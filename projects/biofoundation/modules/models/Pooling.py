from torch import nn, randn, Tensor, mean
from torch.nn import functional as F

class AttentivePooling(nn.Module):
    # taken from OG paper: https://github.com/apple/ml-aim/blob/main/aim-v1/aim/v1/torch/layers.py
    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        num_queries: int = 1,
        use_batch_norm: bool = True,
        qkv_bias: bool = False,
        average_pool: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.average_pool = average_pool
 
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.cls_token = nn.Parameter(randn(1, num_queries, dim) * 0.02)
        # self.bn = (
        #     nn.BatchNorm1d(dim, affine=False, eps=1e-6)
        #     if use_batch_norm
        #     else nn.Identity()
        # )
 
    def forward(self, x: Tensor) -> Tensor:
        x = x[:, 1:, :]  # exclude the ViT CLS token (could also be done in code later if neccessary)
        B, N, C = x.shape
        #x = self.bn(x.transpose(-2, -1)).transpose(-2, -1) #done with fc_norm later
        cls_token = self.cls_token.expand(B, -1, -1)
 
        q = cls_token.reshape(
            B, self.num_queries, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
 
        x_cls = F.scaled_dot_product_attention(q, k, v)
        x_cls = x_cls.transpose(1, 2).reshape(B, self.num_queries, C)
        x_cls = x_cls.mean(dim=1) if self.average_pool else x_cls
        return x_cls
    
class AveragePooling(nn.Module):
    def __init__(self):
        super(AveragePooling, self).__init__()
        # No parameters to initialize for average pooling

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, N, embed_dim) where the first token is the CLS token.
        Returns:
            Pooled features of shape (B, embed_dim).
        """
        # Exclude the CLS token and work on patch tokens only.
        x_patch = x[:, 1:, :]  # shape: (B, N-1, embed_dim)
        
        # Apply average pooling across the sequence dimension (N-1)
        pooled = mean(x_patch, dim=1)  # shape: (B, embed_dim)
        return pooled

class AttentivePooling_old(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Args:
            embed_dim: Dimension of the patch embeddings.
            num_heads: Number of attention heads.
        """
        super(AttentivePooling_old, self).__init__()
        # Using torch's built-in multihead attention
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Learnable query parameter, shape: (1, 1, embed_dim)
        # This query will be repeated for each sample in the batch.
        self.query = nn.Parameter(randn(1, 1, embed_dim))
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, N, embed_dim) where the first token is the CLS token.
        Returns:
            Pooled features of shape (B, embed_dim).
        """
        # Exclude the CLS token and work on patch tokens only.
        x_patch = x[:, 1:, :]  # shape: (B, N-1, embed_dim)
        B = x_patch.shape[0]
        
        # Expand the learnable query for each instance in the batch.
        # Query shape becomes: (B, 1, embed_dim)
        query_expanded = self.query.expand(B, -1, -1)
        
        # Apply multihead attention:
        # Query: (B, 1, embed_dim)
        # Key, Value: (B, N-1, embed_dim)
        # Output shape: (B, 1, embed_dim)
        attn_output, attn_weights = self.mha(query_expanded, x_patch, x_patch)
        
        # Squeeze the sequence dimension to obtain (B, embed_dim)
        pooled = attn_output.squeeze(1)
        return pooled


