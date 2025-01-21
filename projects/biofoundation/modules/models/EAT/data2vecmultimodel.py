# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from typing import Callable
from functools import partial

import lightning as L

from .fairseq.ema import EMAModule

from .base import (
    MaskSeed,
    ModalitySpecificEncoder,
    get_annealed_rate,
)

from .modules import (
    AltBlock,
    Decoder1d,
)

from .images import ImageEncoder

logger = logging.getLogger(__name__)


class Data2VecMultiModel(nn.Module):
    def make_modality_encoder(
        self,
        cfg,
        embed_dim: int,
        make_block: Callable[[float], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
    ) -> ModalitySpecificEncoder:
        enc_cls = ImageEncoder

        return enc_cls(cfg, embed_dim, make_block, norm_layer, layer_norm_first)

    def __init__(self, multimodel, modality, skip_ema=False, task=None):
        super().__init__()
        self.modality = modality
        self.cfg = multimodel
        cfg = multimodel
        self.task = task

        make_layer_norm = partial(
            nn.LayerNorm, eps=cfg.norm_eps, elementwise_affine=cfg.norm_affine
        )

        def make_block(drop_path, dim=None, heads=None):
            return AltBlock(
                cfg.embed_dim if dim is None else dim,
                cfg.num_heads if heads is None else heads,
                cfg.mlp_ratio,
                qkv_bias=True,
                drop=cfg.encoder_dropout,
                attn_drop=cfg.attention_dropout,
                mlp_drop=cfg.activation_dropout,
                post_mlp_drop=cfg.post_mlp_drop,
                drop_path=drop_path,
                norm_layer=make_layer_norm,
                layer_norm_first=cfg.layer_norm_first,
                ffn_targets=not cfg.end_of_block_targets,
            )

        # extract CNN encoder and CNN decoder from modified data2vec image modality (see image.py)
        self.modality_encoder = self.make_modality_encoder(
            self.modality,
            cfg.embed_dim,
            make_block,
            make_layer_norm,
            cfg.layer_norm_first,
        )

        self.ema = None

        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale

        self.dropout_input = nn.Dropout(cfg.dropout_input)

        dpr = np.linspace(cfg.start_drop_path_rate, cfg.end_drop_path_rate, cfg.depth)

        self.blocks = nn.ModuleList([make_block(dpr[i]) for i in range(cfg.depth)])

        self.norm = None
        if cfg.layer_norm_first:
            self.norm = make_layer_norm(cfg.embed_dim)

        if self.cfg.mae_init:
            self.apply(self._init_weights)
        else:
            from .fairseq.transformer_sentence_encoder import init_bert_params

            self.apply(init_bert_params)

        self.modality_encoder.reset_parameters()

        # make teacher model
        if not skip_ema:
            self.ema = self.make_ema_teacher(cfg.ema_decay, cfg)

            self.recon_proj = None
            if cfg.recon_loss > 0:
                self.recon_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim // 3)

        for pn, p in self.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias"):
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}
            if cfg.decoder_group and "decoder" in pn:
                p.param_group = "decoder"

        self.num_updates = 0

    def _init_weights(self, m):

        try:
            from apex.normalization import FusedLayerNorm

            fn = FusedLayerNorm
        except:
            fn = nn.LayerNorm

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, fn):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    @torch.no_grad()
    def make_ema_teacher(self, ema_decay, ema_config):
        ema_config.ema_decay = ema_decay
        ema_config.ema_fp32 = True
        ema_config.log_norms = self.cfg.log_norms
        ema_config.add_missing_params = False

        model_copy = self.make_target_model()

        return EMAModule(
            model_copy,
            ema_config,
            copy_model=False,
        )

    # teacher model (with independent CNN encoder and Transformer encoder)
    def make_target_model(self):
        logger.info("making target model")

        model_copy = Data2VecMultiModel(
            self.cfg, self.modality, skip_ema=True, task=self.task
        )

        if self.cfg.ema_encoder_only:
            model_copy = model_copy.blocks
            for p_s, p_t in zip(self.blocks.parameters(), model_copy.parameters()):
                p_t.data.copy_(p_s.data)
        else:
            for p_s, p_t in zip(self.parameters(), model_copy.parameters()):
                p_t.data.copy_(p_s.data)

            for mod_enc in model_copy.modality_encoders.values():
                mod_enc.decoder = None
                if not mod_enc.modality_cfg.ema_local_encoder:
                    mod_enc.local_encoder = None
                    mod_enc.project_features = None

        model_copy.requires_grad_(False)
        return model_copy

    # teacher model updated with EMA
    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.ema is not None and (
            (self.num_updates == 0 and num_updates > 1)
            or self.num_updates >= num_updates
        ):
            pass
        elif self.training and self.ema is not None:
            ema_weight_decay = None
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay, weight_decay=ema_weight_decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.blocks if self.cfg.ema_encoder_only else self)

        self.num_updates = num_updates

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        k = prefix + "_ema"
        if self.ema is not None:
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        elif k in state_dict:
            del state_dict[k]

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg, task=task, skip_ema=cfg.skip_ema)

    def forward(
        self,
        source,
        target=None,
        id=None,
        mode=None,
        padding_mask=None,
        mask=True,
        features_only=False,
        force_remove_masked=False,
        remove_extra_tokens=True,
        precomputed_mask=None,
    ):
        mask_seeds = None
        if id is not None:
            mask_seeds = MaskSeed(seed=self.cfg.seed, update=self.num_updates, ids=id)

        # extract (unmasked) features using a CNN encoder
        extractor_out = self.modality_encoder(
            source,
            padding_mask,
            mask,
            remove_masked=not features_only or force_remove_masked,
            clone_batch=self.cfg.clone_batch if not features_only else 1,
            mask_seeds=mask_seeds,
            precomputed_mask=precomputed_mask,
        )

        # x in shape ( batch_size * clone batch, patch_frame(64) * patch_freqency(8) * unmask_ratio(0.2) + 1(cls_token), 768(feature dimension) )
        # EAT does not employ the ablibi mechanism in Transformer
        x = extractor_out["x"]
        encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]

        if self.dropout_input is not None:
            x = self.dropout_input(x)

        # standard Transformer (for student encoder)
        layer_results = []
        for i, blk in enumerate(self.blocks):
            if (
                not self.training
                or self.cfg.layerdrop == 0
                or (np.random.random() > self.cfg.layerdrop)
            ):
                x, lr = blk(x, padding_mask=masked_padding_mask)
                if features_only:
                    layer_results.append(lr)

        if self.norm is not None:
            x = self.norm(x)

        # extract features for fine-tuning
        if features_only:
            if remove_extra_tokens:
                x = x[:, self.modality_encoder.modality_cfg.num_extra_tokens :]
                if masked_padding_mask is not None:
                    masked_padding_mask = masked_padding_mask[
                        :, self.modality_encoder.modality_cfg.num_extra_tokens :
                    ]

            return {
                "x": x,
                "padding_mask": masked_padding_mask,
                "layer_results": layer_results,
                "mask": encoder_mask,
            }

        # decode features merged with masked tokens, dx in shape (batch_size * clone_batch, patch, 768)
        dx = self.forward_decoder(
            x,
            self.modality_encoder,
            self.modality_encoder.decoder,
            encoder_mask,
        )

        p = next(self.ema.model.parameters())
        device = x.device
        dtype = x.dtype
        ema_device = p.device
        ema_dtype = p.dtype

        if not self.cfg.ema_same_dtype:
            dtype = ema_dtype

        if ema_device != device or ema_dtype != dtype:
            logger.info(f"adjusting ema dtype to {dtype} and device to {device}")
            self.ema.model = self.ema.model.to(dtype=dtype, device=device)
            ema_dtype = dtype

            def to_device(d):
                for k, p in d.items():
                    if isinstance(d[k], dict):
                        to_device(d[k])
                    else:
                        d[k] = p.to(device=device)

            to_device(self.ema.fp32_params)
        tm = self.ema.model

        # encode audio spectrogram using teacher model
        with torch.no_grad():
            tm.eval()

            if self.cfg.ema_encoder_only:
                assert target is None
                ema_input = extractor_out["local_features"]
                ema_input = self.modality_encoder.contextualized_features(
                    ema_input.to(dtype=ema_dtype),
                    padding_mask,
                    mask=False,
                    remove_masked=False,
                )
                ema_blocks = tm
            else:
                ema_blocks = tm.blocks
                if self.modality_encoder.modality_cfg.ema_local_encoder:
                    inp = (
                        target.to(dtype=ema_dtype)
                        if target is not None
                        else source.to(dtype=ema_dtype)
                    )
                    ema_input = tm.modality_encoders[mode](
                        inp,
                        padding_mask,
                        mask=False,
                        remove_masked=False,
                    )
                else:
                    assert target is None
                    ema_input = extractor_out["local_features"]
                    ema_feature_enc = tm.modality_encoders[mode]
                    ema_input = ema_feature_enc.contextualized_features(
                        ema_input.to(dtype=ema_dtype),
                        padding_mask,
                        mask=False,
                        remove_masked=False,
                    )

            ema_padding_mask = ema_input["padding_mask"]
            ema_input = ema_input["x"]

            # extract target features using teacher CNN encoder
            # ema_input in shape (batch_size, patch + 1(cls_token), feature_dimension)
            y = []
            ema_x = []
            extra_tokens = self.modality_encoder.modality_cfg.num_extra_tokens
            for i, blk in enumerate(ema_blocks):
                ema_input, lr = blk(ema_input, padding_mask=ema_padding_mask)
                y.append(lr[:, extra_tokens:])
                ema_x.append(ema_input[:, extra_tokens:])

        # EAT utilize total 12 Transformer block layer output average as target
        y = self.make_targets(y, self.average_top_k_layers)
        orig_targets = y

        # multiply the target value according to the number of clone batch
        if self.cfg.clone_batch > 1:
            y = y.repeat_interleave(self.cfg.clone_batch, 0)

        # extract values in masked position to make prediction
        masked = encoder_mask.mask.unsqueeze(-1)
        masked_b = encoder_mask.mask.bool()
        y = y[masked_b]

        if dx.size(1) == masked_b.size(1):
            dx = dx[masked_b]
        else:
            dx = dx.reshape(-1, dx.size(-1))

        sample_size = masked.sum().long()

        result = {
            "losses": {},
            "sample_size": sample_size,
        }

        sample_size = result["sample_size"]  # TODO: Isnt that doubled?

        # EAT employ utterance-level loss by using mean pooling in patch dimension
        if self.cfg.cls_loss > 0:
            assert extra_tokens > 0
            cls_target = orig_targets.mean(dim=1)
            if self.cfg.clone_batch > 1:
                cls_target = cls_target.repeat_interleave(self.cfg.clone_batch, 0)
            cls_pred = x[:, extra_tokens - 1]

            result["losses"]["cls"] = self.d2v_loss(cls_pred, cls_target) * (
                self.cfg.cls_loss * sample_size
            )

        if self.cfg.recon_loss > 0:

            with torch.no_grad():
                target = self.modality_encoder.patchify(source)  # (btz,1,512,16*16)
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.0e-6) ** 0.5  # (btz,1,512,1)

                if self.cfg.clone_batch > 1:
                    target = target.repeat_interleave(
                        self.cfg.clone_batch, 0
                    )  # (btz*clone_btz,1,512,1)

                if masked_b is not None:
                    target = target[masked_b]

            recon = dx
            if self.recon_proj is not None:
                recon = self.recon_proj(recon)

            result["losses"]["recon"] = (
                self.d2v_loss(recon, target.float()) * self.cfg.recon_loss
            )

        if self.cfg.d2v_loss > 0:
            reg_loss = self.d2v_loss(dx, y)
            result["losses"]["d2v"] = reg_loss * self.cfg.d2v_loss

        return result

    def forward_decoder(
        self,
        x,
        feature_extractor,
        decoder,
        mask_info,
    ):
        x = feature_extractor.decoder_input(x, mask_info)
        x = decoder(*x)

        return x

    # That is the loss function for both utterance and Frame Loss
    def d2v_loss(self, x, y):
        x = x.view(-1, x.size(-1)).float()
        y = y.view(-1, x.size(-1))

        if self.loss_beta == 0:
            loss = F.mse_loss(x, y, reduction="none")
        else:
            loss = F.smooth_l1_loss(x, y, reduction="none", beta=self.loss_beta)

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(x.size(-1))

        reg_loss = loss * scale

        return reg_loss

    # Average top-k layers output from teacher model to provide a target for the student
    def make_targets(self, y, num_layers):

        with torch.no_grad():
            target_layer_results = y[-num_layers:]

            permuted = False
            if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BTC -> BCT
                ]
                permuted = True
            if self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    F.batch_norm(
                        tl.float(), running_mean=None, running_var=None, training=True
                    )
                    for tl in target_layer_results
                ]
            if self.cfg.instance_norm_target_layer:
                target_layer_results = [
                    F.instance_norm(tl.float()) for tl in target_layer_results
                ]
            if permuted:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
                ]
            if self.cfg.layer_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-1:])
                    for tl in target_layer_results
                ]

        y = target_layer_results[0].float()
        for tl in target_layer_results[1:]:
            y.add_(tl.float())
        y = y.div_(len(target_layer_results))

        if self.cfg.layer_norm_targets:
            y = F.layer_norm(y, y.shape[-1:])

        if self.cfg.instance_norm_targets:
            y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)

        return y

    def remove_pretrain_components(self):
        self.ema = None
        self.shared_decoder = None
        self.modality_encoder.decoder = None
        self.recon_proj = None
