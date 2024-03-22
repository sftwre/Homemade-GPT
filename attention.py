import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):


    def __init__(self, d_in, d_out, block_size, dropout, num_heads, qkv_bias=False) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size), diagonal=1))

    def forward(self, x):
         
         b, num_tokens, d_in = x.shape

         queries = self.W_query(x)
         keys = self.W_key(x)
         values = self.W_value(x)
         
         # split weights
         queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
         keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
         values = values.view(b, num_tokens, self.num_heads, self.head_dim)

         # flip num_tokens and num_heads -> (b, num_heads, num_tokens, head_dim)
         queries = queries.transpose(1, 2)
         keys = keys.transpose(1, 2)
         values = values.transpose(1, 2)

         # compute causal masked attention with dropout

         # (b, num_heads, num_tokens, head_dim) x (b, num_heads, head_dim, num_tokens)
         attn_scores = queries @ keys.transpose(2, 3)
         mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
         mask_unsqueezed = mask_bool.unsqueeze(0).unsqueeze(0)
         attn_scores.masked_fill_(mask_unsqueezed, -torch.inf)

         attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
         attn_weights = self.dropout(attn_weights)

         context_vec = (attn_weights @ values).transpose(1, 2)
         context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
         context_vec = self.out_proj(context_vec)
         return context_vec
