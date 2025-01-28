# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple
from typing import Optional

from .fairseq.data_utils import compute_mask_indices
from .fairseq.grad_multiply import GradMultiply
from .fairseq.utils import index_put


logger = logging.getLogger(__name__)


MaskSeed = namedtuple("MaskSeed", ["seed", "update", "ids"])
MaskInfo = namedtuple("MaskInfo", ["x_unmasked", "mask", "ids_restore", "ids_keep"])


class ModalitySpecificEncoder(nn.Module):
    def __init__(
        self,
        modality_cfg,
        embed_dim: int,
        local_encoder: nn.Module,
        project_features: nn.Module,
        fixed_positional_encoder: Optional[nn.Module],
        relative_positional_encoder: Optional[nn.Module],
        context_encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()

        self.modality_cfg = modality_cfg
        self.local_encoder = local_encoder
        self.project_features = project_features
        self.fixed_positional_encoder = fixed_positional_encoder
        self.relative_positional_encoder = relative_positional_encoder
        self.context_encoder = context_encoder

        self.decoder = decoder

        self.local_grad_mult = self.modality_cfg.local_grad_mult

        self.extra_tokens = None
        if modality_cfg.num_extra_tokens > 0:
            self.extra_tokens = nn.Parameter(
                torch.zeros(1, modality_cfg.num_extra_tokens, embed_dim)
            )
            if not modality_cfg.init_extra_token_zero:
                nn.init.normal_(self.extra_tokens)
            elif self.extra_tokens.size(1) > 1:
                nn.init.normal_(self.extra_tokens[:, 1:])

    def convert_padding_mask(self, x, padding_mask):
        return padding_mask

    def decoder_input(self, x, mask_info: MaskInfo):
        inp_drop = self.modality_cfg.decoder.input_dropout
        if inp_drop > 0:
            x = F.dropout(x, inp_drop, training=self.training, inplace=True)

        num_extra = self.modality_cfg.num_extra_tokens

        if mask_info is not None:
            num_masked = mask_info.ids_restore.shape[1] - x.shape[1] + num_extra

            mask_tokens = x.new_empty(
                x.size(0),
                num_masked,
                x.size(-1),
            ).normal_(0, self.modality_cfg.mask_noise_std)

            x_ = torch.cat([x[:, num_extra:], mask_tokens], dim=1)
            x = torch.gather(x_, dim=1, index=mask_info.ids_restore)

            if self.modality_cfg.decoder.add_positions_masked:
                assert self.fixed_positional_encoder is not None
                pos = self.fixed_positional_encoder(x, None)
                x = x + (pos * mask_info.mask.unsqueeze(-1))
        else:
            x = x[:, num_extra:]

        if self.modality_cfg.decoder.add_positions_all:
            assert self.fixed_positional_encoder is not None
            x = x + self.fixed_positional_encoder(x, None)

        return x, mask_info

    def local_features(self, features):
        if self.local_grad_mult > 0:
            if self.local_grad_mult == 1.0:
                x = self.local_encoder(features)
            else:
                x = GradMultiply.apply(
                    self.local_encoder(features), self.local_grad_mult
                )
        else:
            with torch.no_grad():
                x = self.local_encoder(features)

        x = self.project_features(x)
        return x

    def contextualized_features(
        self,
        x,
        padding_mask,
        mask,
        remove_masked,
        clone_batch: int = 1,
        mask_seeds: Optional[torch.Tensor] = None,
        precomputed_mask=None,
    ):

        if padding_mask is not None:
            padding_mask = self.convert_padding_mask(x, padding_mask)

        local_features = x
        if mask and clone_batch == 1:
            local_features = local_features.clone()

        orig_B, orig_T, _ = x.shape
        pre_mask_B = orig_B
        mask_info = None

        x_pos = None
        if self.fixed_positional_encoder is not None:
            x = x + self.fixed_positional_encoder(x, padding_mask)[:, : x.size(1), :]

        if mask:
            if clone_batch > 1:
                x = x.repeat_interleave(clone_batch, 0)
                if mask_seeds is not None:
                    clone_hash = [
                        int(hash((mask_seeds.seed, ind)) % 1e10)
                        for ind in range(clone_batch - 1)
                    ]
                    clone_hash = torch.tensor([0] + clone_hash).long().view(1, -1)

                    id = mask_seeds.ids
                    id = id.repeat_interleave(clone_batch, 0)
                    id = id.view(-1, clone_batch) + clone_hash.to(id)
                    id = id.view(-1)
                    mask_seeds = MaskSeed(
                        seed=mask_seeds.seed, update=mask_seeds.update, ids=id
                    )
                if padding_mask is not None:
                    padding_mask = padding_mask.repeat_interleave(clone_batch, 0)

            x, mask_info = self.compute_mask(
                x,
                padding_mask,
                mask_seed=mask_seeds,
                apply=self.relative_positional_encoder is not None or not remove_masked,
                precomputed_mask=precomputed_mask,
            )

        if self.relative_positional_encoder is not None:
            x_pos = self.relative_positional_encoder(x)

        masked_padding_mask = padding_mask
        if mask and remove_masked:
            x = mask_info.x_unmasked
            if x_pos is not None:
                x = x + gather_unmasked(x_pos, mask_info)

            if padding_mask is not None and padding_mask.any():
                masked_padding_mask = gather_unmasked_mask(padding_mask, mask_info)
                if not masked_padding_mask.any():
                    masked_padding_mask = None
            else:
                masked_padding_mask = None

        elif x_pos is not None:
            x = x + x_pos

        if self.extra_tokens is not None:
            num = self.extra_tokens.size(1)
            x = torch.cat([self.extra_tokens.expand(x.size(0), -1, -1), x], dim=1)
            if masked_padding_mask is not None:
                # B x T
                masked_padding_mask = F.pad(masked_padding_mask, (num, 0))

        x = self.context_encoder(x, masked_padding_mask)

        return {
            "x": x,
            "local_features": local_features,
            "padding_mask": masked_padding_mask,
            "encoder_mask": mask_info,
        }

    def forward(
        self,
        features,
        padding_mask,
        mask: bool,
        remove_masked: bool,
        clone_batch: int = 1,
        mask_seeds: Optional[torch.Tensor] = None,
        precomputed_mask=None,
    ):
        x = self.local_features(features)
        return self.contextualized_features(
            x,
            padding_mask,
            mask,
            remove_masked,
            clone_batch,
            mask_seeds,
            precomputed_mask,
        )

    def reset_parameters(self):
        pass

    def compute_mask(
        self,
        x,
        padding_mask,
        mask_seed: Optional[MaskSeed],
        apply,
        precomputed_mask,
    ):
        if precomputed_mask is not None:
            mask = precomputed_mask
            mask_info = self.make_maskinfo(x, mask)
        else:
            B, T, C = x.shape
            cfg = self.modality_cfg

            mask_prob = cfg.mask_prob

            if (
                cfg.mask_prob_min is not None
                and cfg.mask_prob_min >= 0
                and cfg.mask_prob_min < mask_prob
            ):
                mask_prob = np.random.uniform(cfg.mask_prob_min, mask_prob)

            if mask_prob > 0:
                if cfg.mask_length == 1:
                    mask_info = random_masking(x, mask_prob, mask_seed)
                else:
                    if self.modality_cfg.inverse_mask:
                        mask_prob = 1 - mask_prob

                    mask = compute_mask_indices(
                        (B, T),
                        padding_mask,
                        mask_prob,
                        cfg.mask_length,
                        min_masks=1,
                        require_same_masks=True,
                        mask_dropout=cfg.mask_dropout,
                        add_masks=cfg.add_masks,
                        seed=mask_seed.seed if mask_seed is not None else None,
                        epoch=mask_seed.update if mask_seed is not None else None,
                        indices=mask_seed.ids if mask_seed is not None else None,
                    )

                    mask = torch.from_numpy(mask)
                    if self.modality_cfg.inverse_mask:
                        mask = 1 - mask
                    mask_info = self.make_maskinfo(x, mask)
            else:
                mask_info = None

        if apply:
            x = self.apply_mask(x, mask_info)

        return x, mask_info

    def make_maskinfo(self, x, mask, shape=None):
        if shape is None:
            B, T, D = x.shape
        else:
            B, T, D = shape

        mask = mask.to(torch.uint8)
        ids_shuffle = mask.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1).unsqueeze(-1).expand(-1, -1, D)

        len_keep = T - mask[0].sum()
        if self.modality_cfg.keep_masked_pct > 0:
            len_keep += round((T - int(len_keep)) * self.modality_cfg.keep_masked_pct)

        ids_keep = ids_shuffle[:, :len_keep]

        if shape is not None:
            x_unmasked = None
        else:
            ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
            x_unmasked = torch.gather(x, dim=1, index=ids_keep)

        mask_info = MaskInfo(
            x_unmasked=x_unmasked,
            mask=mask,
            ids_restore=ids_restore,
            ids_keep=ids_keep,
        )
        return mask_info

    def apply_mask(self, x, mask_info):
        cfg = self.modality_cfg
        B, T, C = x.shape

        if mask_info is not None:
            mask = mask_info.mask
            if cfg.encoder_zero_mask:
                x = x * (1 - mask.type_as(x).unsqueeze(-1))
            else:
                num_masks = mask.sum().item()
                masks = x.new_empty(num_masks, x.size(-1)).normal_(
                    0, cfg.mask_noise_std
                )
                x = index_put(x, mask, masks)
        if cfg.mask_channel_prob > 0:
            mask_channel = compute_mask_indices(
                (B, C),
                None,
                cfg.mask_channel_prob,
                cfg.mask_channel_length,
            )
            mask_channel = torch.from_numpy(mask_channel).unsqueeze(1).expand(-1, T, -1)
            x = index_put(x, mask_channel, 0)
        return x

    def remove_pretraining_modules(self, keep_decoder=False):
        if not keep_decoder:
            self.decoder = None


def get_annealed_rate(start, end, curr_step, total_steps):
    if curr_step >= total_steps:
        return end
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


# adapted from MAE
def random_masking(x, mask_ratio, mask_seed: Optional[MaskSeed]):
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    generator = None
    if mask_seed is not None:
        seed = int(
            hash((mask_seed.seed, mask_seed.update, mask_seed.ids.sum().item())) % 1e6
        )
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)

    noise = torch.rand(N, L, generator=generator, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = noise.argsort(dim=1)  # ascend: small is keep, large is remove
    ids_restore = ids_shuffle.argsort(dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
    x_unmasked = torch.gather(x, dim=1, index=ids_keep)

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], dtype=x.dtype, device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    ids_restore = ids_restore.unsqueeze(-1).expand(-1, -1, D)

    return MaskInfo(
        x_unmasked=x_unmasked, mask=mask, ids_restore=ids_restore, ids_keep=ids_keep
    )


def gather_unmasked(x: torch.Tensor, mask_info: MaskInfo) -> torch.Tensor:
    return torch.gather(
        x,
        dim=1,
        index=mask_info.ids_keep,
    )


def gather_unmasked_mask(x: torch.Tensor, mask_info: MaskInfo) -> torch.Tensor:
    return torch.gather(
        x,
        dim=1,
        index=mask_info.ids_keep[..., 0],  # ignore the feature dimension
    )
