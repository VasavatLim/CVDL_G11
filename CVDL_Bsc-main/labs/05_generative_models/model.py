from enum import Enum
from queue import LifoQueue

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# helper functions
# --------------------------------------------------------------------------------------
def get_timestep_embedding(timesteps: torch.Tensor, time_emb_dim: int) -> torch.Tensor:
    if timesteps.ndim == 0:
        timesteps = timesteps.unsqueeze(0)
    half = time_emb_dim // 2
    frequencies = torch.exp(
        -torch.log(torch.tensor(10_000))
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    angles = timesteps[:, None].float() * frequencies[None, :]
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


# --------------------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------------------
class DConvBlock(nn.Module):
    def __init__(
        self, in_channel: int, out_channel: int, time_emb_dim: int, drop_p=0.0
    ):
        super(DConvBlock, self).__init__()
        self.norm1 = nn.GroupNorm(32, in_channel, eps=1e-05, affine=True)
        self.norm2 = nn.GroupNorm(32, out_channel, eps=1e-05, affine=True)
        self.dropout = nn.Dropout(p=drop_p, inplace=False)
        self.nonlinearity = nn.SiLU()
        self.conv1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=3, stride=1, padding=1
        )
        self.time_emb = nn.Linear(
            in_features=time_emb_dim, out_features=out_channel, bias=True
        )

    def forward(self, x, t_emb):
        # Input conv
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        # Time modulation
        if t_emb is not None:
            t_hidden = self.time_emb(self.nonlinearity(t_emb))
            x = x + t_hidden[:, :, None, None]
        # Output conv
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self, in_channel: int, out_channel: int, time_emb_dim: int, drop_p=0.0
    ):
        super(ResBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.time_emb_dim = time_emb_dim
        self.drop_p = drop_p

    def forward(self, x, t_emb):
        # TODO implement Residual Block with time embedding
        pass


class AttBlock(nn.Module):
    def __init__(self, channels):
        super(AttBlock, self).__init__()
        self.in_channels = channels

        self.norm = nn.GroupNorm(32, channels, eps=1e-05, affine=True)
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        # Normalize input features
        x_normalized = self.norm(x)

        # Generate query, key, value tensors
        q = self.q_proj(x_normalized)
        k = self.k_proj(x_normalized)
        v = self.v_proj(x_normalized)

        # Reshape and compute attention scores
        b, c, h, w = q.shape
        q = q.view(b, c, h * w).permute(0, 2, 1)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)
        attention_scores = torch.bmm(q, k)

        # Scale attention scores
        attention_scores = attention_scores / (c**0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Attend to values and reshape to original dimensions
        attended_values = torch.bmm(v, attention_weights.transpose(1, 2))
        attended_values = attended_values.view(b, c, h, w)

        # Project output and add residual
        projected_output = self.proj_out(attended_values)
        return x + projected_output


class PerformerAttBlock(nn.Module):
    def __init__(self, channels):
        super(PerformerAttBlock, self).__init__()
        self.channels = channels
        self.num_features = channels // 2
        self.norm = nn.GroupNorm(32, channels, eps=1e-05, affine=True)

        # Projections for queries, keys, and values
        self.q_proj = nn.Linear(channels, self.num_features)
        self.k_proj = nn.Linear(channels, self.num_features)
        self.v_proj = nn.Linear(channels, channels)

        self.orth_matrix = self.create_orthogonal_random_features()

    def create_orthogonal_random_features(self):
        # Create a random matrix and orthogonalize it
        random_matrix = torch.randn(self.num_features, self.num_features)
        q, _ = torch.linalg.qr(random_matrix)
        return q

    def forward(self, x):
        b, c, h, w = x.shape
        x_normalized = self.norm(x)
        x_flat = x_normalized.view(b, c, h * w).permute(0, 2, 1)  # (b, hw, c)

        # apply projections
        q = self.q_proj(x_flat)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        # use orthogonal features
        self.orth_matrix = self.orth_matrix.to(x.device)
        if self.orth_matrix.device != x.device:
            self.orth_matrix = self.orth_matrix.to(x.device)
        q = q @ self.orth_matrix
        k = k @ self.orth_matrix

        # normalization and softmax approximation
        d_k = self.num_features**0.5
        q = q / (d_k + 1e-6)
        k = k / (d_k + 1e-6)

        # compute attention
        kv = torch.einsum("bhc,bhd->bcd", k, v)
        qkv = torch.einsum("bhc,bcd->bhd", q, kv)

        # reshape and add residual connection
        attended_values = qkv.permute(0, 2, 1).view(b, c, h, w)
        return x + attended_values


class UpSample(nn.Module):
    def __init__(self, channels: int, scale_factor=2, mode="nearest"):
        super(UpSample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, channels: int):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        return self.conv(x)


class TimeModulatedSequential(nn.ModuleList):
    def forward(self, x, t_emb):
        for module in self:
            if isinstance(module, TimeModulatedSequential):
                x = module(x, t_emb)
            elif isinstance(module, DConvBlock):
                x = module(x, t_emb)
            elif isinstance(module, ResBlock):
                x = module(x, t_emb)
            else:
                x = module(x)
        return x


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class LVLBLOCK(Enum):
    res = ResBlock
    cov = DConvBlock


class ATTBLOCK(Enum):
    self = AttBlock
    perf = PerformerAttBlock


class SDUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels_base: int,
        multipliers: tuple[int, ...],
        attn_levels: tuple[int, ...],
        n_lvlblocks: int,
        t_emb_dim: int,
        lvlblock_type: str,
        attblock_type: str,
    ) -> None:
        super(SDUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels_base = channels_base
        self.multipliers = multipliers
        self.n_lvlblocks = n_lvlblocks
        self.t_emb_dim = t_emb_dim

        # helper vars
        level_channels = [channels_base * mult for mult in multipliers]
        nlevels = len(multipliers)
        cur_channels = channels_base

        # blocks
        lvlblock = LVLBLOCK[lvlblock_type].value
        attblock = ATTBLOCK[attblock_type].value

        # time embdedding
        self.time_embedding = nn.Sequential(
            nn.Linear(channels_base, t_emb_dim, bias=True),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim, bias=True),
        )

        # input
        self.input = nn.Conv2d(
            in_channels=in_channels,
            out_channels=cur_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # down blocks
        self.d_blocks = nn.ModuleList()
        for i_level in range(nlevels):
            level_block = TimeModulatedSequential()
            for _ in range(n_lvlblocks):
                level_block.append(
                    lvlblock(
                        in_channel=cur_channels,
                        out_channel=level_channels[i_level],
                        time_emb_dim=t_emb_dim,
                    )
                )
                if i_level in attn_levels:
                    level_block.append(
                        attblock(
                            channels=level_channels[i_level],
                        )
                    )
                cur_channels = level_channels[i_level]
            self.d_blocks.append(level_block)
            self.d_blocks.append(DownSample(level_channels[i_level]))

        # mid blocks
        self.m_blocks = TimeModulatedSequential()
        for _ in range(n_lvlblocks):
            self.m_blocks.append(
                lvlblock(
                    in_channel=cur_channels,
                    out_channel=cur_channels,
                    time_emb_dim=t_emb_dim,
                )
            )

        # up blocks
        self.u_blocks = nn.ModuleList()
        for i_level in reversed(range(nlevels)):
            level_block = TimeModulatedSequential()
            self.u_blocks.append(UpSample(cur_channels))
            cur_channels = cur_channels + level_channels[i_level]
            for _ in range(n_lvlblocks):
                level_block.append(
                    lvlblock(
                        in_channel=cur_channels,
                        out_channel=level_channels[i_level],
                        time_emb_dim=t_emb_dim,
                    )
                )
                if i_level in attn_levels:
                    level_block.append(
                        attblock(
                            channels=level_channels[i_level],
                        )
                    )
                cur_channels = level_channels[i_level]
            self.u_blocks.append(level_block)

        # output
        self.output = nn.Sequential(
            nn.GroupNorm(32, cur_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=cur_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # time embedding
        t_emb = get_timestep_embedding(t, self.channels_base).to(x.dtype)
        t_emb = self.time_embedding(t_emb)

        # input
        x = self.input(x)

        skip = LifoQueue()
        # down
        for module in self.d_blocks:
            if isinstance(module, TimeModulatedSequential):
                x = module(x, t_emb)
                skip.put(x)
            else:
                x = module(x)
        # middle
        x = self.m_blocks(x, t_emb)
        # up
        for module in self.u_blocks:
            if isinstance(module, TimeModulatedSequential):
                x = torch.cat((skip.get(), x), dim=1)
                x = module(x, t_emb)
            else:
                x = module(x)

        # output
        return self.output(x)
