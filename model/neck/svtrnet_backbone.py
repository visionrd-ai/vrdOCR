
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


trunc_normal_ = lambda x: nn.init.trunc_normal_(x, std=0.02)
normal_ = lambda x: nn.init.normal_(x)
zeros_ = lambda x: nn.init.constant_(x, 0.0)
ones_ = lambda x: nn.init.constant_(x, 1.0)



def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # create broadcasting shape
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x / keep_prob * random_tensor
    return output


class ConvBNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias_attr=False,
        groups=1,
        act=nn.GELU,
    ):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias_attr,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMixer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        HW=[8, 25],
        local_k=[3, 3],
    ):
        super(ConvMixer, self).__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2d(
            dim, dim, local_k, 1, padding=[local_k[0] // 2, local_k[1] // 2], groups=num_heads
        )

    def forward(self, x):
        h, w = self.HW
        x = x.permute(0, 2, 1).reshape(-1, self.dim, h, w)
        x = self.local_mixer(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        mixer="Global",
        HW=None,
        local_k=[7, 11],
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        self.mixer = mixer

        if mixer == "Local" and HW is not None:
            H, W = HW
            hk, wk = local_k
            mask = torch.ones(H * W, H + hk - 1, W + wk - 1, dtype=torch.float32)
            for h in range(H):
                for w in range(W):
                    mask[h * W + w, h: h + hk, w: w + wk] = 0.0
            mask = mask[:, hk // 2: H + hk // 2, wk // 2: W + wk // 2].flatten(1)
            mask_inf = torch.full([H * W, H * W], float('-inf'), dtype=torch.float32)
            mask = torch.where(mask < 1, mask, mask_inf)
            self.mask = mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.mixer == "Local":
            attn += self.mask.to(attn.device)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mixer="Global",
        local_mixer=[7, 11],
        HW=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        epsilon=1e-6,
        prenorm=True,
    ):
        super(Block, self).__init__()
        self.prenorm = prenorm
        self.norm1 = norm_layer(dim, eps=epsilon)
        self.mixer = Attention(
            dim,
            num_heads=num_heads,
            mixer=mixer,
            HW=HW,
            local_k=local_mixer,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        ) if mixer in ["Global", "Local"] else ConvMixer(dim, num_heads=num_heads, HW=HW, local_k=local_mixer)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim, eps=epsilon)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=[32, 100],
        in_channels=3,
        embed_dim=768,
        sub_num=2,
        patch_size=[4, 4],
        mode="pope",
    ):
        super(PatchEmbed, self).__init__()
        num_patches = (img_size[1] // (2**sub_num)) * (img_size[0] // (2**sub_num))
        self.img_size = img_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        if mode == "pope":
            if sub_num == 2:
                self.proj = nn.Sequential(
                    ConvBNLayer(in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1, act=nn.GELU),
                    ConvBNLayer(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, act=nn.GELU),
                )
            elif sub_num == 1:
                self.proj = nn.Sequential(
                    ConvBNLayer(in_channels, embed_dim, kernel_size=3, stride=2, padding=1, act=nn.GELU)
                )
            elif sub_num == 0:
                self.proj = nn.Sequential(
                    ConvBNLayer(in_channels, embed_dim, kernel_size=1, stride=1, padding=0, act=nn.GELU)
                )
        elif mode == "linear":
            self.proj = nn.Linear(3 * patch_size[0] * patch_size[1], embed_dim)

    def forward(self, x):
        x = self.proj(x)
        return x


class SVTRNet(nn.Module):
    def __init__(
        self,
        img_size=[32, 100],
        in_channels=3,
        embed_dim=768,
        num_heads=8,
        depth=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        epsilon=1e-6,
    ):
        super(SVTRNet, self).__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, in_channels=in_channels, embed_dim=embed_dim)
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mixer="Global",
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer,
                    epsilon=epsilon,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim, eps=epsilon)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


# Example usage
if __name__ == "__main__":
    model = SVTRNet()
    # Generate a random input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(2, 3, 32, 100)  # Example for a batch of 2 images
    output = model(input_tensor)
    print("Output shape:", output.shape)