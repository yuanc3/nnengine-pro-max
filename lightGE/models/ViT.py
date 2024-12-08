from lightGE.core.nn import Model, Linear, Sequential, ReLU, LeakyReLU, Dropout, LayerNorm, ModuleList
import numpy as np
from lightGE.core.tensor import Tensor


class Transformer(Model):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super(Transformer, self).__init__()
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)))
            self.layers.append(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x    
        return x

class PreNorm(Model):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn
    def forward(self, x):
        return self.fn(self.norm(x))

class Attention(Model):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        self.dim_heads = dim_head
        self.heads = heads
        self.scale = Tensor(dim_head ** -0.5)
        
        self.to_q = Linear(dim, inner_dim, bias = False)
        self.to_k = Linear(dim, inner_dim, bias = False)
        self.to_v = Linear(dim, inner_dim, bias = False)
        
        self.to_out = Sequential([
            Linear(inner_dim, dim),
            Dropout(dropout)
            ]
        ) 

    def forward(self, x):
        # b, n, hd = x.shape
        # sp = (b, self.heads, n, self.dim_heads)
        q = self.to_q(x).rearrange('b n (h d) -> b h n d', h = self.heads, d = self.dim_heads)
        k = self.to_k(x).rearrange('b n (h d) -> b h n d', h = self.heads, d = self.dim_heads)
        v = self.to_v(x).rearrange('b n (h d) -> b h n d', h = self.heads, d = self.dim_heads)
        dots = q.mm(k.transpose((0, 1, 3, 2))) * self.scale
        attn = dots.softmax()
        out = attn.mm(v)
        
        out = out.rearrange('b h n d -> b n (h d)', h = self.heads)
        return self.to_out(out)

class FeedForward(Model):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward, self).__init__()
        self.net = Sequential([
            Linear(dim, hidden_dim), 
            LeakyReLU(),
            Dropout(dropout),
            Linear(hidden_dim, dim),
            Dropout(dropout)
            ])
    def forward(self, x):
        return self.net(x)
    
class ViT(Model):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super(ViT, self).__init__()
        image_height, image_width = (image_size,image_size)
        self.patch_height, self.patch_width = (patch_size,patch_size)
        assert image_height % self.patch_height == 0 and image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // self.patch_height) * (image_width // self.patch_width)
        patch_dim = channels * self.patch_height * self.patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.to_patch_embedding = Linear(patch_dim, dim)
        self.pos_embedding = Tensor(np.random.random((1, num_patches + 1, dim)), autograd=True)
        self.cls_token = Tensor(np.random.random((1, 1, dim)), autograd=True)
        self.dropout = Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.mlp_head = Sequential([
            LayerNorm(dim),
            Linear(dim, num_classes)],                                
        )
        
    def forward(self, img):
        b, c, h, w = img.shape
        pattern = f'b c (hi p1) (wi p2) -> b (hi wi) (p1 p2 c)'
        x = img.rearrange(pattern, 
                          hi = h//self.patch_height, 
                          wi = w//self.patch_width, 
                          p1 = self.patch_height, 
                          p2 = self.patch_width)
        x = self.to_patch_embedding(x) 
        b, n, d = x.shape
        # '() n d -> b n d'
        cls_token, _= self.cls_token.broadcast((b, 1, d))
        
        x = cls_token.concat(x, axis=1)
        x = x + self.pos_embedding
        x = self.dropout(x) 
        x = self.transformer(x) 
        x = x.mean(1) if self.pool == 'mean' else x.cls().squeeze(1)
        x = self.mlp_head(x)

        return x.softmax().log()
