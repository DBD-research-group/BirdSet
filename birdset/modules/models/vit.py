from pathlib import Path
from lightning import LightningModule
import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import PatchEmbed
from timm.models.vision_transformer import VisionTransformer
from functools import partial

class VIT(LightningModule, VisionTransformer):

    def __init__(self, 
                 img_size_x,
                 img_size_y,
                 patch_size,
                 in_chans,
                 embed_dim,
                 global_pool,
                 norm_layer,
                 mlp_ratio,
                 qkv_bias,
                 eps,
                 drop_path,
                 num_heads,
                 depth,
                 target_length,
                 num_classes,
                 pretrained_weights_path: str,
    ):
        
        LightningModule.__init__(self)
        
        VisionTransformer.__init__(
            self,
            img_size = (img_size_x, img_size_y),
            patch_size = patch_size,
            in_chans = in_chans,
            embed_dim = embed_dim,
            depth = depth,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            qkv_bias = qkv_bias,
            norm_layer = partial(nn.LayerNorm, eps=eps),
            num_classes = num_classes,
            drop_path_rate=drop_path,
        )
        
        self.img_size = (img_size_x, img_size_y)
        self.global_pool = global_pool

        norm_layer = partial(nn.LayerNorm, eps=eps)
        self.fc_norm = norm_layer(embed_dim)

        self.pretrained_weights_path =  pretrained_weights_path
        self.target_length = target_length
        self.load_pretrained_weights()

    def load_pretrained_weights(self): 
        img_size = (self.target_length, self.img_size[1])

        num_patches = 512 

        self.patch_embed = PatchEmbed(img_size, 16, 1, 768)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, 768), requires_grad=False) #to load pretrained pos embed
        
        pretrained_state_dict = torch.load(self.pretrained_weights_path, map_location="cpu")["model"]

        for k in ['head.weight', 'head.bias']:
            if k in pretrained_state_dict and pretrained_state_dict[k].shape != self.state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del pretrained_state_dict[k]
        
        self.load_state_dict(pretrained_state_dict, strict=False)

        patch_hw = (img_size[1] // 16, img_size[0] // 16) # 16=patchsize
        pos_embed = self.get_2d_sincos_pos_embed_flexible(self.pos_embed.size(-1), patch_hw, cls_token=True) # not trained, overwrite from sincos
        self.pos_embed.data = torch.from_numpy(pos_embed).float().unsqueeze(0)

    def get_2d_sincos_pos_embed_flexible(self, embed_dim, grid_size, cls_token=False):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """
        grid_h = np.arange(grid_size[0], dtype=np.float32)
        grid_w = np.arange(grid_size[1], dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        return pos_embed


    def get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
        return emb
    
    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out) # (M, D/2)
        emb_cos = np.cos(out) # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) # batch, patch, embed
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)        

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, input_values, **kwargs):
        input_values = self.forward_features(input_values)
        pred = self.head(input_values)
        return pred 