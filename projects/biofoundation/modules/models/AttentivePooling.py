from torch import nn, randn



class AttentivePooling(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Args:
            embed_dim: Dimension of the patch embeddings.
            num_heads: Number of attention heads.
        """
        super(AttentivePooling, self).__init__()
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