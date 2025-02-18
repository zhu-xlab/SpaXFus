import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from pdb import set_trace as stx
import numbers
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange
import math
from typing import Optional, Callable
from einops import rearrange, repeat
from functools import partial

NEG_INF = -1000000

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=16):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)

class SCAB(nn.Module):
    def __init__(self, num_feat):
        super(SCAB, self).__init__()

        squeeze_factor=1

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)

##########################################################################
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
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

class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops

#########################################
class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W, mask=None):
        group_size = (H, W)
        B_, N, C = x.shape
        assert H * W == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1).contiguous()  # (B_, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()  # (2h-1)*(2w-1) 2

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

            pos = self.pos(biases)  # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nP = mask.shape[0]
            attn = attn.view(B_ // nP, nP, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0)  # (B, nP, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SSM(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class SXM(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SSM(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3,1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x

class XIM(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.self_attention = SSM(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.ln_1 = norm_layer(hidden_dim)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = SCAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input):
        # x [B,C,5]
        B, C, L = input.shape
        input = input.view(B,C,1,L).contiguous()  # [B,C,1,5]
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3,1).contiguous()
        x = x.view(B, C,L).contiguous()
        return x

##########################################################################
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x

##########################################################################
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x

class CAttention(nn.Module):
    def __init__(self, in_planes):
        super(CAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        weight = avg_out + max_out
        out = torch.mul(x, self.sigmoid(weight))
        return out

class ResXblock(nn.Module):
    def __init__(self,in_channels,mid=64,out_channels=2,groups=4):
        super(ResXblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid, 1, bias=True,groups=groups)
        self.conv2 = self.convlayer(mid,mid,3,groups)
        self.conv3 = nn.Conv2d(mid, out_channels, 1, bias=True,groups=groups)
        self.Lrelu=nn.GELU()

    def convlayer(self, in_channels, out_channels, kernel_size,groups):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True,groups=groups)
        Lrelu = nn.GELU()
        layers = filter(lambda x: x is not None, [pader, conver, Lrelu])
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.Lrelu(self.conv1(x))
        out2=self.conv2(out1)
        residuals = self.conv3(out2)
        out=residuals+x
        return self.Lrelu(out)

def compute_channel_sta(tensor):
    b, c, h, w = tensor.shape
    max_pool = torch.amax(tensor, dim=(2, 3))

    mean_pool = torch.mean(tensor, dim=(2, 3))

    min_pool = torch.amin(tensor, dim=(2, 3))

    std_dev = torch.std(tensor, dim=(2, 3))

    def compute_entropy(x):
        eps = 1e-9
        probs = x / x.sum(dim=(2, 3), keepdim=True)
        probs = probs.clamp(min=eps)
        entropy = -probs * torch.log(probs)
        return entropy.sum(dim=(2, 3))

    entropy = compute_entropy(tensor)

    # features = torch.stack([max_pool, mean_pool, min_pool, std_dev, entropy], dim=2)
    features = torch.stack([max_pool, mean_pool, min_pool, std_dev], dim=2)
    return features

class MDDLStage(nn.Module):
    def __init__(self,mamban,hrms_channels,lrms_channels,mambamid,groups,down='maxpooling',up='bilinear',ratio=4):
        """down: maxpooling,avgpooling, conv, nearest,bilibear,bicubic,
            up: CARAFE, deconv, bilinear,bicubic,pixelshuffle
        """
        super(MDDLStage, self).__init__()
        self.up=up
        self.channeln = lrms_channels
        self.down=down
        self.HR_Fi2 = self.convlayer(lrms_channels, hrms_channels,  1,1)
        self.HR_Fi2t = self.convlayer(hrms_channels, lrms_channels, 1, 1)
        self.HR_Fi2t.add_module('transResX', ResXblock(lrms_channels, mambamid, lrms_channels, groups))
        self.HRe=CAttention(lrms_channels)
        self.MHe=CAttention(lrms_channels)

        if down=='maxpooling':
            self.LR_down=nn.MaxPool2d(ratio)
        elif down=='avgpooling':
            self.LR_down=nn.AvgPool2d(ratio)
        elif down=='conv':
            self.LR_down=self.convlayer(lrms_channels,out_channels,3,ratio)
        elif down=='nearest':
            self.LR_down=nn.Upsample(scale_factor=1/ratio, mode='nearest')
        elif down=='bilinear':
            self.LR_down=nn.Upsample(scale_factor=1/ratio, mode='bilinear')
        elif down=='bicubic':
            self.LR_down=nn.Upsample(scale_factor=1/ratio, mode='bicubic')
        else:
            print('Without this type of downsampling!')

        if up=='bilinear':
            self.LR_up=nn.Upsample(scale_factor=ratio,mode='bilinear')
        elif up=='bicubic':
            self.LR_up=nn.Upsample(scale_factor=ratio,mode='bicubic')

        self.LRe = CAttention(lrms_channels)
        self.MLe=CAttention(lrms_channels)
        self.mlp_ratio =2.0
        self.drop_path_rate=0.
        base_d_state = 4
        self.priorex=self.convlayer(lrms_channels,mambamid, 1, 1)
        self.SpaXIPrior = nn.ModuleList([
            SXM(
                hidden_dim=mambamid,
                drop_path=self.drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
            )
            for i in range(mamban)])
        self.priorfus = self.convlayer(mambamid,lrms_channels,  1, 1)
        sta_number=4
        self.smamba=XIM(
                hidden_dim=sta_number,
                drop_path=self.drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
            )
        self.smamba_w = nn.Conv1d(in_channels=sta_number, out_channels=1, kernel_size=1)
        self.Prioreu=CAttention(lrms_channels)
        self.Xkeu=CAttention(lrms_channels)

    def convlayer(self, in_channels, out_channels, kernel_size,stride):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=True)
        Lrelu = nn.GELU()
        layers = filter(lambda x: x is not None, [pader, conver, Lrelu])
        return nn.Sequential(*layers)

    def deconvlayer(self, in_channels, out_channels,stride):
        conver = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=stride, stride=stride, padding=0, bias=True)
        Lrelu = nn.GELU()
        layers = filter(lambda x: x is not None, [conver, Lrelu])
        return nn.Sequential(*layers)

    def forward(self, X,Ml,Mh):
        euXk=self.Xkeu(X)
        Xkfi2=self.HR_Fi2(X)
        Xkfi2fi2t=self.HR_Fi2t(Xkfi2)
        Mhfi2t=self.HR_Fi2t(Mh)
        eMhfi2t=self.MHe(Mhfi2t)
        eXkfi2fi2t=self.HRe(Xkfi2fi2t)
        DXk=self.LR_down(X)
        DtDXk=self.LR_up(DXk)
        DtMl = self.LR_up(Ml)
        eDtDXk=self.LRe(DtDXk)
        eDtMl=self.MLe(DtMl)
        Zk =self.priorex(X)
        B, C, H, W = Zk.shape
        Zk = rearrange(Zk, "b c h w -> b (h w) c").contiguous()
        for layer in self.SpaXIPrior:
            Zk = layer(Zk, [H, W])
        Zk = rearrange(Zk, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        Zk=self.priorfus(Zk)
        Zk_sta=compute_channel_sta(Zk)
        Smamba=self.smamba(Zk_sta)
        Smamba_conv = self.smamba_w(Smamba.permute(0, 2, 1) )
        Zk = Smamba_conv.view(B, self.channeln, 1, 1)* Zk

        euZk=self.Prioreu(Zk)
        Xnew=X-euXk+eDtMl-eDtDXk+eMhfi2t-eXkfi2fi2t+euZk
        return Xnew

class HR_Net(nn.Module):
    def __init__(self,hrms_channels, mid,lrms_channels,groups):
        super(HR_Net, self).__init__()
        self.HR_Fi2t = self.convlayer(hrms_channels, lrms_channels, 1, 1)
        self.HR_Fi2t.add_module('transResX', ResXblock(lrms_channels, mid, lrms_channels, groups))
        self.HRe = CAttention(lrms_channels)

    def convlayer(self, in_channels, out_channels, kernel_size,stride):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=True)
        Lrelu = nn.GELU()
        layers = filter(lambda x: x is not None, [pader, conver, Lrelu])
        return nn.Sequential(*layers)

    def forward(self, x):
        Mhfi2t = self.HR_Fi2t(x)
        eMhfi2t=self.HRe(Mhfi2t)
        return eMhfi2t

class LR_Net(nn.Module):
    def __init__(self,lrms_channels,up='bilinear',ratio=4):
        super(LR_Net, self).__init__()
        if up=='CARAFE':
            self.LR_up=CARAFE(lrms_channels,ratio)
        elif up=='deconv':
            self.LR_up=self.deconvlayer(lrms_channels,lrms_channels,ratio)
        elif up=='bilinear':
            self.LR_up=nn.Upsample(scale_factor=ratio,mode='bilinear')
        elif up=='bicubic':
            self.LR_up=nn.Upsample(scale_factor=ratio,mode='bicubic')
        elif up == 'pixelshuffle':
            self.LR_up=self.convlayer(lrms_channels,lrms_channels*ratio*ratio,3,1)
            self.LR_up.add_module('ps', nn.PixelShuffle(ratio))
        else:
            print('Without this type of Upsampling!')
        self.LRe = CAttention(lrms_channels)

    def convlayer(self, in_channels, out_channels, kernel_size,stride):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=True)
        Lrelu = nn.GELU()
        layers = filter(lambda x: x is not None, [pader, conver, Lrelu])
        return nn.Sequential(*layers)

    def deconvlayer(self, in_channels, out_channels,stride):
        conver = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=stride, stride=stride, padding=0, bias=True)
        Lrelu = nn.GELU()
        layers = filter(lambda x: x is not None, [conver, Lrelu])
        return nn.Sequential(*layers)

    def forward(self, x):
        DtMl=self.LR_up(x)
        eDtMl=self.LRe(DtMl)
        return eDtMl
    
from model.ODconv import ODConv2d

class SpaXFus(nn.Module):
    def __init__(self, mamban,hrms_channels,lrms_channels,mid,groups,down='maxpooling',up='bilinear',ratio=4):
        super(SpaXFus, self).__init__()
        self.IRN_HR=HR_Net(hrms_channels, mid,lrms_channels,groups)
        self.IRN_LR=LR_Net(lrms_channels,up=up,ratio=ratio)
        lambdh = torch.randn((1),requires_grad=True)
        lambdl = torch.randn((1),requires_grad=True)
        self.lambdhr = torch.nn.Parameter(lambdh)
        self.lambdlr = torch.nn.Parameter(lambdl)
        nn.init.constant_(self.lambdhr, 1.)
        nn.init.constant_(self.lambdlr, 1.)
        self.fusion = self.convlayer(2 * lrms_channels, lrms_channels, 1, 1)
        self.OS1 = MDDLStage(mamban,hrms_channels,lrms_channels,mid,groups,down,up,ratio)
        self.OS2 = MDDLStage(mamban,hrms_channels,lrms_channels,mid,groups,down,up,ratio)
        self.OS3 = MDDLStage(mamban,hrms_channels,lrms_channels,mid,groups,down,up,ratio)
        self.OS4 = MDDLStage(mamban,hrms_channels,lrms_channels,mid,groups,down,up,ratio)
        self.OS5 = MDDLStage(mamban,hrms_channels,lrms_channels,mid,groups,down,up,ratio)
        self.OS6 = MDDLStage(mamban,hrms_channels,lrms_channels,mid,groups,down,up,ratio)
        self.OS7 = MDDLStage(mamban,hrms_channels,lrms_channels,mid,groups,down,up,ratio)
        self.OS8 = MDDLStage(mamban,hrms_channels,lrms_channels,mid,groups,down,up,ratio)
        self.OS9 = MDDLStage(mamban,hrms_channels,lrms_channels,mid,groups,down,up,ratio)
        self.fus=ODConv2d(lrms_channels*9,lrms_channels,1)

    def convlayer(self, in_channels, out_channels, kernel_size,stride):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=True)
        Lrelu = nn.GELU()
        layers = filter(lambda x: x is not None, [pader, conver, Lrelu])
        return nn.Sequential(*layers)

    def forward(self,Mh,Ml):
        X0_HR = self.IRN_HR(Mh)
        X0_LR = self.IRN_LR(Ml)
        X = torch.cat([X0_HR, X0_LR], 1)
        X0 = self.fusion(X)
        X1 = self.OS1(X0, Ml, Mh)
        X2 = self.OS2(X1, Ml, Mh)
        X3 = self.OS3(X2, Ml, Mh)
        X4 = self.OS4(X3, Ml, Mh)
        X5 = self.OS5(X4, Ml, Mh)
        X6 = self.OS6(X5, Ml, Mh)
        X7 = self.OS7(X6, Ml, Mh)
        X8 = self.OS8(X7, Ml, Mh)
        X9 = self.OS9(X8, Ml, Mh)
        X_fus = self.fus(torch.cat([X1, X2,X3,X4,X5,X6,X7,X8,X9], 1))
        return X_fus


class TFMamba(nn.Module):
    def __init__(self, lrchannel, hrchannel, ratio, num_blocks=[2, 3, 3, 4, 4], mlp_ratio=2., drop_path_rate=0.):
        super(TFMamba, self).__init__()
        self.mlp_ratio = mlp_ratio
        base_d_state = 4
        self.up = nn.Upsample(scale_factor=ratio, mode='bicubic')
        self.encoder1_pan = nn.Sequential(
            nn.Conv2d(in_channels=hrchannel,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )
        self.encoder2_pan = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.encoder1_lr = nn.Sequential(
            nn.Conv2d(in_channels=lrchannel,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU())
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )

        self.fusion2 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=2,
                      stride=2),
            nn.PReLU(),
        )

        self.restore1 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      ),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=2,
                               stride=2),
            nn.PReLU())

        self.restore2 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=2,
                               stride=2),
            nn.PReLU()
        )

        self.restore3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )

        self.res1mamba = nn.ModuleList([
            SXM(
                hidden_dim=128,
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
            )
            for i in range(num_blocks[2])])

        self.final=nn.Sequential(
            nn.Conv2d(in_channels=64,
                  out_channels=lrchannel,
                  kernel_size=3,
                  stride=1,
                  padding=1),
            nn.Tanh()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_pan, x_lr):
        x_lr = self.up(x_lr)
        encoder1_pan = self.encoder1_pan(x_pan)
        encoder1_lr = self.encoder1_lr(x_lr)

        encoder2_pan = self.encoder2_pan(encoder1_pan)
        encoder2_lr = self.encoder2_lr(encoder1_lr)

        fusion1 = self.fusion1(torch.cat((encoder2_pan, encoder2_lr), dim=1))

        # _, _, H, W = fusion1.shape
        # fusion1 = rearrange(fusion1, "b c h w -> b (h w) c").contiguous()
        # for layer in self.fus1mamba:
        #     fusion1 = layer(fusion1, [H, W])
        # fusion1 = rearrange(fusion1, "b (h w) c -> b c h w", h=H, w=W).contiguous()

        fusion2 = self.fusion2(fusion1)

        # _, _, H, W = fusion2.shape
        # fusion2 = rearrange(fusion2, "b c h w -> b (h w) c").contiguous()
        # for layer in self.fus2mamba:
        #     fusion2 = layer(fusion2, [H, W])
        # fusion2 = rearrange(fusion2, "b (h w) c -> b c h w", h=H, w=W).contiguous()

        restore1 = self.restore1(fusion2)

        _, _, H, W = restore1.shape
        restore1 = rearrange(restore1, "b c h w -> b (h w) c").contiguous()
        for layer in self.res1mamba:
            restore1 = layer(restore1, [H, W])
        restore1 = rearrange(restore1, "b (h w) c -> b c h w", h=H, w=W).contiguous()

        restore2 = self.restore2(torch.cat((restore1, fusion1), dim=1))

        # _, _, H, W = restore2.shape
        # restore2 = rearrange(restore2, "b c h w -> b (h w) c").contiguous()
        # for layer in self.res2mamba:
        #     restore2 = layer(restore2, [H, W])
        # restore2 = rearrange(restore2, "b (h w) c -> b c h w", h=H, w=W).contiguous()

        restore3 = self.restore3(torch.cat((restore2, encoder1_lr, encoder1_pan), dim=1))

        # _, _, H, W = restore3.shape
        # restore3 = rearrange(restore3, "b c h w -> b (h w) c").contiguous()
        # for layer in self.res3mamba:
        #     restore3 = layer(restore3, [H, W])
        # restore3 = rearrange(restore3, "b (h w) c -> b c h w", h=H, w=W).contiguous()

        return self.final(restore3)

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    height = 128
    width = 128
    hrimg=torch.randn((1, 1, height, width)).cuda()
    lrimg=torch.randn((1, 4, 32, 32)).cuda()
    # model=TFMamba(4,1,4,num_blocks=[4, 6, 6, 4, 8]).cuda()
    model =Mamba4ever(4,1,4,64,4).cuda()
    import numpy as np
    s = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Number of params: %d' % s)

    y = model(hrimg,lrimg)
    print(y.shape)