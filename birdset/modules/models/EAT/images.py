# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Callable, Optional
from timm.models.layers import to_2tuple

from .mae import get_2d_sincos_pos_embed_flexible, PatchEmbed_new


from .base import (
    ModalitySpecificEncoder,
    MaskSeed,
)
from .modules import (
    BlockEncoder,
    Decoder2d,
    FixedPositionalEncoder,
)



class ImageEncoder(ModalitySpecificEncoder):
    def __init__(
        self,
        modality_cfg,
        embed_dim: int,
        make_block: Callable[[float, Optional[int], Optional[int]], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool
    ):
        
        img_size = (modality_cfg.target_length,modality_cfg.target_height)
        patch_size = to_2tuple(modality_cfg.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.H = img_size[0] // patch_size[0]
        self.W = img_size[1] // patch_size[1]
        self.hw = (self.H,self.W)

        local_encoder = PatchEmbed_new(
            img_size,
            modality_cfg.patch_size,
            modality_cfg.in_chans,
            modality_cfg.embed_dim,
        )

        # CNN initialize
        w = local_encoder.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if modality_cfg.embed_dim != embed_dim:
            local_encoder = nn.Sequential(
                local_encoder,
                nn.Linear(modality_cfg.embed_dim, embed_dim),
            )

        project_features = nn.Identity()

        # note: max_length control the maximum time length of audio -> "64" for 10s, here we define it as 2min, you can change it yourself
        max_length = modality_cfg.max_length
        pos_embed = nn.Parameter(
            torch.zeros(1, max_length*self.W, embed_dim), requires_grad=False
        )

        # side_n = int(num_patches ** 0.5)
        # note: we fix the variable length sequence problem here -> support up to 2min audio 
        emb = get_2d_sincos_pos_embed_flexible(
            pos_embed.shape[-1],
            (max_length,self.W),  
            cls_token=False,
        )
        
        pos_embed.data.copy_(torch.from_numpy(emb[:max_length*self.W,:]).float().unsqueeze(0)) 
        fixed_positional_encoder = (
            FixedPositionalEncoder(pos_embed) if modality_cfg.fixed_positions else None
        )

        dpr = np.linspace(
            modality_cfg.start_drop_path_rate,
            modality_cfg.end_drop_path_rate,
            modality_cfg.prenet_depth,
        )

        context_encoder = BlockEncoder(
            nn.ModuleList(make_block(dpr[i]) for i in range(modality_cfg.prenet_depth)),
            norm_layer(embed_dim) if not layer_norm_first else None,
            layer_norm_first,
            modality_cfg.prenet_layerdrop,
            modality_cfg.prenet_dropout,
        )

        decoder = (
            Decoder2d(modality_cfg.decoder, embed_dim, self.H, self.W)
        )

        super().__init__(
            modality_cfg=modality_cfg,
            embed_dim=embed_dim,
            local_encoder=local_encoder,
            project_features=project_features,
            fixed_positional_encoder=fixed_positional_encoder,
            relative_positional_encoder=None,
            context_encoder=context_encoder,
            decoder=decoder
        )

    def reset_parameters(self):
        super().reset_parameters()
        if self.decoder is not None:
            self.decoder.reset_parameters()

    @torch.no_grad()
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)   audio: (N,1,H,W)   1024/16 = 64   128/16 = 8
        x: (N, L, patch_size**2 *3)
        """
        if self.modality_cfg.in_chans == 1:
            p = self.modality_cfg.patch_size
            h = imgs.shape[2] // p
            w = imgs.shape[3] // p
            #h,w = self.patch_embed.patch_hw
            x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
            
        else:
            p = self.modality_cfg.patch_size
            h = w = imgs.shape[2] // p
            x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
            x = torch.einsum("nchpwq->nhwpqc", x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))

        return x

    @torch.no_grad()
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.modality_cfg.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def compute_mask(
        self,
        x,
        padding_mask,
        mask_seed: Optional[MaskSeed],
        apply,
        shape=None,
        precomputed_mask=None,
    ):
        mlen = self.modality_cfg.mask_length
        if mlen <= 1:
            return super().compute_mask(
                x, padding_mask, mask_seed, apply, precomputed_mask
            )

        if precomputed_mask is not None:
            mask = precomputed_mask
        else:
            from .fairseq.data_utils import compute_block_mask_2d

            if shape is not None:
                B, L, D = shape
            else:
                B, L, D = x.shape

            mask = compute_block_mask_2d(
                shape=(B, L),
                mask_prob=self.modality_cfg.mask_prob,
                mask_length=self.modality_cfg.mask_length,
                mask_prob_adjust=self.modality_cfg.mask_prob_adjust,
                inverse_mask=self.modality_cfg.inverse_mask,
                require_same_masks=True,
                mask_dropout=self.modality_cfg.mask_dropout
            )

            #TODO: Put mask on same device as x if not already
            if mask.device != x.device:
                mask = mask.to(x.device)
            

        mask_info = self.make_maskinfo(x, mask, shape)
        if apply:
            x = self.apply_mask(x, mask_info)

        return x, mask_info

    def decoder_input(self, x, mask_info):
        return super().decoder_input(x, mask_info)