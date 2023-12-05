import torch
import torch.nn as nn
import torch.nn.functional as F

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def freeze_module(module):
    if isinstance(module,nn.Module):
        print(f'Freezing {type(module)} ...')
        for name,param in module.named_parameters():
            param.requires_grad = False
        module.eval()
        module.train = disabled_train
    elif isinstance(module,nn.Parameter):
        print(f'Freezing {type(module)} ...')
        module.requires_grad = True
    else:
        raise TypeError

class MLPHead(nn.Module):
    def __init__(self,in_dim,out_dim,drop_rate=0.):
        super(MLPHead,self).__init__()
        self.dense = nn.Linear(in_dim,in_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.out_proj = nn.Linear(in_dim,out_dim)

    def forward(self,x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, query_dim, kv_dim, num_heads, output_dim=None):
        super(MultiHeadCrossAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.output_dim = output_dim if output_dim else embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(kv_dim, embed_dim)
        self.v_proj = nn.Linear(kv_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        q = self.q_proj(query) # NLC
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape and transpose for multi-head attention
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        # Combine heads
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)

        # Final linear projection
        output = self.out_proj(context)
        return output

class TextAttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, txt_dim:int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.cross_attn = MultiHeadCrossAttention(
            embed_dim=embed_dim,
            query_dim=txt_dim,
            kv_dim=embed_dim,
            num_heads=num_heads,
            output_dim=output_dim
        )
        self.num_heads = num_heads

    def forward(self, x, txt_feat):
        # import pdb;pdb.set_trace()
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x = x.permute(1,0,2).contiguous() # N(HW+1)C
        txt_feat = txt_feat.unsqueeze(dim=1)  # NC -> N(1)C
        x = self.cross_attn(query=txt_feat,key=x,value=x) # N(1)C
        return x.squeeze(dim=1)

def interpolate_pos_embed(pos_embed_ckpt,input_resolution=512):
    embedding_size = 2048
    orig_size = 224//32
    new_size = input_resolution//32
    num_extra_tokens = 1 # mean token

    if orig_size != new_size:
        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        extra_tokens = pos_embed_ckpt[:num_extra_tokens,:]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_ckpt[num_extra_tokens:,:]
        pos_tokens = pos_tokens.reshape(orig_size, orig_size, embedding_size).permute(2, 0, 1)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens.unsqueeze(dim=0),
            size=(new_size, new_size),
            mode='bicubic',
            align_corners=False
        )
        pos_tokens = pos_tokens.squeeze(dim=0).permute(1, 2, 0).flatten(0,1)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
        return new_pos_embed
    else:
        return pos_embed_ckpt