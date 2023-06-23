import math
import torch
import torchvision
import numpy as np
from torch import nn
from torchtools.utils import Diffuzz


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class GlobalResponseNorm(nn.Module):  # from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class AttnBlock(nn.Module):
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0):
        super().__init__()
        self.self_attn = self_attn
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        self.attention = Attention2D(c, nhead, dropout)
        self.kv_mapper = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_cond, c)
        )

    def forward(self, x, kv):
        kv = self.kv_mapper(kv)
        x = x + self.attention(self.norm(x), kv, self_attn=self.self_attn)
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, c, dropout=0.0):
        super().__init__()
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            nn.Linear(c, c * 4),
            nn.GELU(),
            GlobalResponseNorm(c * 4),
            nn.Dropout(dropout),
            nn.Linear(c * 4, c)
        )

    def forward(self, x):
        x = x + self.channelwise(self.norm(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class TimestepBlock(nn.Module):
    def __init__(self, c, c_timestep):
        super().__init__()
        self.mapper = nn.Linear(c_timestep, c * 2)

    def forward(self, x, t):
        a, b = self.mapper(t)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + a) + b


class Attention2D(nn.Module):
    def __init__(self, c, nhead, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(c, nhead, dropout=dropout, bias=True, batch_first=True)

    def forward(self, x, kv, self_attn=False):
        orig_shape = x.shape
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)  # Bx4xHxW -> Bx(HxW)x4
        if self_attn:
            kv = torch.cat([x, kv], dim=1)
        x = self.attn(x, kv, kv, need_weights=False)[0]
        x = x.permute(0, 2, 1).view(*orig_shape)
        return x

class ResBlockStageB(nn.Module):
    def __init__(self, c, c_skip=None, kernel_size=3, dropout=0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(c, c, kernel_size=kernel_size, padding=kernel_size//2, groups=c)
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            nn.Linear(c+c_skip, c*4),
            nn.GELU(),
            GlobalResponseNorm(c*4),
            nn.Dropout(dropout),
            nn.Linear(c*4, c)
        )

    def forward(self, x, x_skip=None):
        x_res = x
        x = self.norm(self.depthwise(x))
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.channelwise(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + x_res


class ResBlock(nn.Module):
    def __init__(self, c, c_skip=0, kernel_size=3, dropout=0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(c + c_skip, c, kernel_size=kernel_size, padding=kernel_size // 2, groups=c)
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            nn.Linear(c, c * 4),
            nn.GELU(),
            GlobalResponseNorm(c * 4),
            nn.Dropout(dropout),
            nn.Linear(c * 4, c)
        )

    def forward(self, x, x_skip=None):
        x_res = x
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.norm(self.depthwise(x)).permute(0, 2, 3, 1)
        x = self.channelwise(x).permute(0, 3, 1, 2)
        return x + x_res


class DiffNeXt(nn.Module):
    def __init__(self, c_in=4, c_out=4, c_r=64, patch_size=2, c_cond=1024, c_hidden=[320, 640, 1280, 1280],
                 nhead=[-1, 10, 20, 20], blocks=[4, 4, 14, 4], level_config=['CT', 'CTA', 'CTA', 'CTA'],
                 inject_effnet=[False, True, True, True], effnet_embd=16, clip_embd=1024, kernel_size=3, dropout=0.1,
                 self_attn=True):
        super().__init__()
        self.c_r = c_r
        self.c_cond = c_cond
        if not isinstance(dropout, list):
            dropout = [dropout] * len(c_hidden)

        # CONDITIONING
        self.clip_mapper = nn.Linear(clip_embd, c_cond)
        self.effnet_mappers = nn.ModuleList([
            nn.Conv2d(effnet_embd, c_cond, kernel_size=1) if inject else None for inject in
            inject_effnet + list(reversed(inject_effnet))
        ])
        self.seq_norm = nn.LayerNorm(c_cond, elementwise_affine=False, eps=1e-6)

        self.embedding = nn.Sequential(
            nn.PixelUnshuffle(patch_size),
            nn.Conv2d(c_in * (patch_size ** 2), c_hidden[0], kernel_size=1),
            LayerNorm2d(c_hidden[0], elementwise_affine=False, eps=1e-6)
        )

        def get_block(block_type, c_hidden, nhead, c_skip=0, dropout=0):
            if block_type == 'C':
                return ResBlockStageB(c_hidden, c_skip, kernel_size=kernel_size, dropout=dropout)
            elif block_type == 'A':
                return AttnBlock(c_hidden, c_cond, nhead, self_attn=self_attn, dropout=dropout)
            elif block_type == 'F':
                return FeedForwardBlock(c_hidden, dropout=dropout)
            elif block_type == 'T':
                return TimestepBlock(c_hidden, c_r)
            else:
                raise Exception(f'Block type {block_type} not supported')

        # BLOCKS
        # -- down blocks
        self.down_blocks = nn.ModuleList()
        for i in range(len(c_hidden)):
            down_block = nn.ModuleList()
            if i > 0:
                down_block.append(nn.Sequential(
                    LayerNorm2d(c_hidden[i - 1], elementwise_affine=False, eps=1e-6),
                    nn.Conv2d(c_hidden[i - 1], c_hidden[i], kernel_size=2, stride=2),
                ))
            for _ in range(blocks[i]):
                for block_type in level_config[i]:
                    c_skip = c_cond if inject_effnet[i] else 0
                    down_block.append(get_block(block_type, c_hidden[i], nhead[i], c_skip=c_skip, dropout=dropout[i]))
            self.down_blocks.append(down_block)

        # -- up blocks
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(c_hidden))):
            up_block = nn.ModuleList()
            for j in range(blocks[i]):
                for k, block_type in enumerate(level_config[i]):
                    c_skip = c_hidden[i] if i < len(c_hidden) - 1 and j == k == 0 else 0
                    c_skip += c_cond if inject_effnet[i] else 0
                    up_block.append(get_block(block_type, c_hidden[i], nhead[i], c_skip=c_skip, dropout=dropout[i]))
            if i > 0:
                up_block.append(nn.Sequential(
                    LayerNorm2d(c_hidden[i], elementwise_affine=False, eps=1e-6),
                    nn.ConvTranspose2d(c_hidden[i], c_hidden[i - 1], kernel_size=2, stride=2),
                ))
            self.up_blocks.append(up_block)

        # OUTPUT
        self.clf = nn.Sequential(
            LayerNorm2d(c_hidden[0], elementwise_affine=False, eps=1e-6),
            nn.Conv2d(c_hidden[0], 2 * c_out * (patch_size ** 2), kernel_size=1),
            nn.PixelShuffle(patch_size),
        )

        # --- WEIGHT INIT ---
        self.apply(self._init_weights)  # General init
        for mapper in self.effnet_mappers:
            if mapper is not None:
                nn.init.normal_(mapper.weight, std=0.02)  # conditionings
        nn.init.normal_(self.clip_mapper.weight, std=0.02)  # conditionings
        nn.init.xavier_uniform_(self.embedding[1].weight, 0.02)  # inputs
        nn.init.constant_(self.clf[1].weight, 0)  # outputs

        # blocks
        for level_block in self.down_blocks + self.up_blocks:
            for block in level_block:
                if isinstance(block, ResBlockStageB) or isinstance(block, FeedForwardBlock):
                    block.channelwise[-1].weight.data *= np.sqrt(1 / sum(blocks))
                elif isinstance(block, TimestepBlock):
                    nn.init.constant_(block.mapper.weight, 0)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb

    def gen_c_embeddings(self, clip):
        clip = self.clip_mapper(clip)
        clip = self.seq_norm(clip)
        return clip

    def _down_encode(self, x, r_embed, effnet, clip):
        level_outputs = []
        for i, down_block in enumerate(self.down_blocks):
            effnet_c = None
            for block in down_block:
                if isinstance(block, ResBlockStageB):
                    if effnet_c is None and self.effnet_mappers[i] is not None:
                        effnet_c = self.effnet_mappers[i](nn.functional.interpolate(
                            effnet.float(), size=x.shape[-2:], mode='bicubic', antialias=True, align_corners=True
                        ))
                    skip = effnet_c if self.effnet_mappers[i] is not None else None
                    x = block(x, skip)
                elif isinstance(block, AttnBlock):
                    x = block(x, clip)
                elif isinstance(block, TimestepBlock):
                    x = block(x, r_embed)
                else:
                    x = block(x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, r_embed, effnet, clip):
        x = level_outputs[0]
        for i, up_block in enumerate(self.up_blocks):
            effnet_c = None
            for j, block in enumerate(up_block):
                if isinstance(block, ResBlockStageB):
                    if effnet_c is None and self.effnet_mappers[len(self.down_blocks) + i] is not None:
                        effnet_c = self.effnet_mappers[len(self.down_blocks) + i](nn.functional.interpolate(
                            effnet.float(), size=x.shape[-2:], mode='bicubic', antialias=True, align_corners=True
                        ))
                    skip = level_outputs[i] if j == 0 and i > 0 else None
                    if effnet_c is not None:
                        if skip is not None:
                            skip = torch.cat([skip, effnet_c], dim=1)
                        else:
                            skip = effnet_c
                    x = block(x, skip)
                elif isinstance(block, AttnBlock):
                    x = block(x, clip)
                elif isinstance(block, TimestepBlock):
                    x = block(x, r_embed)
                else:
                    x = block(x)
        return x

    def forward(self, x, r, effnet, clip, x_cat=None, eps=1e-3, return_noise=True):
        if x_cat is not None:
            x = torch.cat([x, x_cat], dim=1)
        # Process the conditioning embeddings
        r_embed = self.gen_r_embedding(r)
        clip = self.gen_c_embeddings(clip)

        # Model Blocks
        x_in = x
        x = self.embedding(x)
        level_outputs = self._down_encode(x, r_embed, effnet, clip)
        x = self._up_decode(level_outputs, r_embed, effnet, clip)
        a, b = self.clf(x).chunk(2, dim=1)
        b = b.sigmoid() * (1 - eps * 2) + eps
        if return_noise:
            return (x_in - a) / b
        else:
            return a, b

    def update_weights_ema(self, src_model, beta=0.999):
        for self_params, src_params in zip(self.parameters(), src_model.parameters()):
            self_params.data = self_params.data * beta + src_params.data * (1 - beta)


class Paella(nn.Module):
    def __init__(self, c_in=256, c_out=256, num_labels=8192, c_r=64, patch_size=2, c_cond=1024,
                 c_hidden=[640, 1280, 1280], nhead=[-1, 16, 16], blocks=[4, 12, 4], level_config=['CT', 'CTA', 'CTA'],
                 effnet_embd=16, byt5_embd=1536, kernel_size=3, dropout=0.1, self_attn=True):
        super().__init__()
        self.c_r = c_r
        self.c_cond = c_cond
        self.num_labels = num_labels
        if not isinstance(dropout, list):
            dropout = [dropout] * len(c_hidden)

        # CONDITIONING
        self.byt5_mapper = nn.Linear(byt5_embd, c_cond)
        self.effnet_mapper = nn.Linear(effnet_embd, c_cond)
        self.seq_norm = nn.LayerNorm(c_cond, elementwise_affine=False, eps=1e-6)

        self.in_mapper = nn.Sequential(
            nn.Embedding(num_labels, c_in),
            nn.LayerNorm(c_in, elementwise_affine=False, eps=1e-6)
        )
        self.embedding = nn.Sequential(
            nn.PixelUnshuffle(patch_size),
            nn.Conv2d(c_in * (patch_size ** 2), c_hidden[0], kernel_size=1),
            LayerNorm2d(c_hidden[0], elementwise_affine=False, eps=1e-6)
        )

        def get_block(block_type, c_hidden, nhead, c_skip=0, dropout=0):
            if block_type == 'C':
                return ResBlock(c_hidden, c_skip, kernel_size=kernel_size, dropout=dropout)
            elif block_type == 'A':
                return AttnBlock(c_hidden, c_cond, nhead, self_attn=self_attn, dropout=dropout)
            elif block_type == 'F':
                return FeedForwardBlock(c_hidden, dropout=dropout)
            elif block_type == 'T':
                return TimestepBlock(c_hidden, c_r)
            else:
                raise Exception(f'Block type {block_type} not supported')

        # BLOCKS
        # -- down blocks
        self.down_blocks = nn.ModuleList()
        for i in range(len(c_hidden)):
            down_block = nn.ModuleList()
            if i > 0:
                down_block.append(nn.Sequential(
                    LayerNorm2d(c_hidden[i - 1], elementwise_affine=False, eps=1e-6),
                    nn.Conv2d(c_hidden[i - 1], c_hidden[i], kernel_size=2, stride=2),
                ))
            for _ in range(blocks[i]):
                for block_type in level_config[i]:
                    down_block.append(get_block(block_type, c_hidden[i], nhead[i], dropout=dropout[i]))
            self.down_blocks.append(down_block)

        # -- up blocks
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(c_hidden))):
            up_block = nn.ModuleList()
            for j in range(blocks[i]):
                for k, block_type in enumerate(level_config[i]):
                    up_block.append(get_block(block_type, c_hidden[i], nhead[i],
                                              c_skip=c_hidden[i] if i < len(c_hidden) - 1 and j == k == 0 else 0,
                                              dropout=dropout[i]))
            if i > 0:
                up_block.append(nn.Sequential(
                    LayerNorm2d(c_hidden[i], elementwise_affine=False, eps=1e-6),
                    nn.ConvTranspose2d(c_hidden[i], c_hidden[i - 1], kernel_size=2, stride=2),
                ))
            self.up_blocks.append(up_block)

        # OUTPUT
        self.clf = nn.Sequential(
            LayerNorm2d(c_hidden[0], elementwise_affine=False, eps=1e-6),
            nn.Conv2d(c_hidden[0], c_out * (patch_size ** 2), kernel_size=1),
            nn.PixelShuffle(patch_size),
        )
        self.out_mapper = nn.Sequential(
            LayerNorm2d(c_out, elementwise_affine=False, eps=1e-6),
            nn.Conv2d(c_out, num_labels, kernel_size=1, bias=False)
        )

        # --- WEIGHT INIT ---
        self.apply(self._init_weights)  # General init

        nn.init.normal_(self.byt5_mapper.weight, std=0.02)  # conditionings
        nn.init.normal_(self.effnet_mapper.weight, std=0.02)
        torch.nn.init.xavier_uniform_(self.embedding[1].weight, 0.02)  # inputs
        nn.init.constant_(self.clf[1].weight, 0)  # outputs
        nn.init.normal_(self.in_mapper[0].weight, std=np.sqrt(1 / num_labels))  # out mapper
        self.out_mapper[-1].weight.data = self.in_mapper[0].weight.data[:, :, None, None].clone()

        # blocks
        for level_block in self.down_blocks + self.up_blocks:
            for block in level_block:
                if isinstance(block, ResBlock) or isinstance(block, FeedForwardBlock):
                    block.channelwise[-1].weight.data *= np.sqrt(1 / sum(blocks))
                elif isinstance(block, TimestepBlock):
                    nn.init.constant_(block.mapper.weight, 0)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb

    def gen_c_embeddings(self, effnet, byt5):
        effnet = effnet.permute(0, 2, 3, 1).view(effnet.size(0), -1, effnet.size(1))
        seq = self.effnet_mapper(effnet)

        if byt5 is not None:
            byt5 = self.byt5_mapper(byt5)
            seq = torch.cat([seq, byt5], dim=1)

        seq = self.seq_norm(seq)
        return seq

    def _down_encode(self, x, r_embed, c_embed):
        level_outputs = []
        for down_block in self.down_blocks:
            for block in down_block:
                if isinstance(block, ResBlock):
                    x = block(x)
                elif isinstance(block, AttnBlock):
                    x = block(x, c_embed)
                elif isinstance(block, TimestepBlock):
                    x = block(x, r_embed)
                else:
                    x = block(x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, r_embed, c_embed):
        x = level_outputs[0]
        for i, up_block in enumerate(self.up_blocks):
            for j, block in enumerate(up_block):
                if isinstance(block, ResBlock):
                    x = block(x, level_outputs[i] if j == 0 and i > 0 else None)
                elif isinstance(block, AttnBlock):
                    x = block(x, c_embed)
                elif isinstance(block, TimestepBlock):
                    x = block(x, r_embed)
                else:
                    x = block(x)
        return x

    def forward(self, x, r, effnet, byt5, x_cat=None):
        if x_cat is not None:
            x = torch.cat([x, x_cat], dim=1)
        # Process the conditioning embeddings
        r_embed = self.gen_r_embedding(r)
        c_embed = self.gen_c_embeddings(effnet, byt5)

        # Model Blocks
        x = self.embedding(self.in_mapper(x).permute(0, 3, 1, 2))
        level_outputs = self._down_encode(x, r_embed, c_embed)
        x = self._up_decode(level_outputs, r_embed, c_embed)
        x = self.out_mapper(self.clf(x))
        return x

    def add_noise(self, x, t, mask=None, random_x=None):
        if mask is None:
            mask = (torch.rand_like(x.float()) <= t[:, None, None]).long()
        if random_x is None:
            random_x = torch.randint_like(x, 0, self.num_labels)
        x = x * (1 - mask) + random_x * mask
        return x, mask

    def get_loss_weight(self, t, mask, min_val=0.3):
        return 1 - (1 - mask) * ((1 - t) * (1 - min_val))[:, None, None]

    def update_weights_ema(self, src_model, beta=0.999):
        for self_params, src_params in zip(self.parameters(), src_model.parameters()):
            self_params.data = self_params.data * beta + src_params.data * (1 - beta)


def sample(model, model_inputs, latent_shape, unconditional_inputs=None, init_x=None, steps=12, renoise_steps=None, temperature = (0.7, 0.3), cfg=(8.0, 8.0), mode = 'multinomial', t_start=1.0, t_end=0.0, sampling_conditional_steps=None, sampling_quant_steps=None, attn_weights=None): # 'quant', 'multinomial', 'argmax'
    device = unconditional_inputs["byt5"].device
    if sampling_conditional_steps is None:
        sampling_conditional_steps = steps
    if sampling_quant_steps is None:
        sampling_quant_steps = steps
    if renoise_steps is None:
        renoise_steps = steps-1
    if unconditional_inputs is None:
        unconditional_inputs = {k: torch.zeros_like(v) for k, v in model_inputs.items()}
    intermediate_images = []
    # with torch.inference_mode():
    init_noise = torch.randint(0, model.num_labels, size=latent_shape, device=device)
    if init_x != None:
        sampled = init_x
    else:
        sampled = init_noise.clone()
    t_list = torch.linspace(t_start, t_end, steps+1)
    temperatures = torch.linspace(temperature[0], temperature[1], steps)
    cfgs = torch.linspace(cfg[0], cfg[1], steps)
    if cfg is not None:
        model_inputs = {k:torch.cat([v, v_u]) for (k, v), (k_u, v_u) in zip(model_inputs.items(), unconditional_inputs.items())}
    for i, tv in enumerate(t_list[:steps]):
        if i >= sampling_quant_steps:
            mode = "quant"
        t = torch.ones(latent_shape[0], device=device) * tv

        if cfg is not None and i < sampling_conditional_steps:
            logits, uncond_logits = model(torch.cat([sampled]*2), torch.cat([t]*2), **model_inputs).chunk(2)
            logits = logits * cfgs[i] + uncond_logits * (1-cfgs[i])
        else:
            logits = model(sampled, t, **model_inputs)

        scores = logits.div(temperatures[i]).softmax(dim=1)

        if mode == 'argmax':
            sampled = logits.argmax(dim=1)
        elif mode == 'multinomial':
            sampled = scores.permute(0, 2, 3, 1).reshape(-1, logits.size(1))
            sampled = torch.multinomial(sampled, 1)[:, 0].view(logits.size(0), *logits.shape[2:])
        elif mode == 'quant':
            sampled = scores.permute(0, 2, 3, 1) @ vqmodel.vquantizer.codebook.weight.data
            sampled = vqmodel.vquantizer.forward(sampled, dim=-1)[-1]
        else:
            raise Exception(f"Mode '{mode}' not supported, use: 'quant', 'multinomial' or 'argmax'")

        intermediate_images.append(sampled)

        if i < renoise_steps:
            t_next = torch.ones(latent_shape[0], device=device) * t_list[i+1]
            sampled = model.add_noise(sampled, t_next, random_x=init_noise)[0]
            intermediate_images.append(sampled)
    return sampled, intermediate_images


class EfficientNetEncoder(nn.Module):
    def __init__(self, c_latent=16, effnet="efficientnet_v2_s"):
        super().__init__()
        if effnet == "efficientnet_v2_s":
            self.backbone = torchvision.models.efficientnet_v2_s(weights='DEFAULT').features.eval()
        else:
            print("Using EffNet L.")
            self.backbone = torchvision.models.efficientnet_v2_l(weights='DEFAULT').features.eval()
        self.mapper = nn.Sequential(
            nn.Conv2d(1280, c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent),  # then normalize them to have mean 0 and std 1
        )

    def forward(self, x):
        return self.mapper(self.backbone(x)).add(1.).div(42.)


class Wrapper(nn.Module):
    def __init__(self, effnet, generator, device=None):
        super().__init__()
        self.effnet = effnet
        self.generator = generator
        self.diffuzz = Diffuzz(device=device)

    def forward(self, x, r, effnet_x, byt5, x_cat=None):
        effnet = self.effnet(effnet_x)
        if np.random.rand() < 0.05:
            effnet = 0 * effnet
        elif np.random.rand() < 0.3:
            effnet = self.noise_effnet_embeds(effnet)
        return self.generator(x, r, effnet, byt5, x_cat)

    def noise_effnet_embeds(self, effnet):
        noise_t = torch.rand(effnet.size(0), device=effnet.device) * 0.5
        effnet = self.diffuzz.diffuse(effnet, noise_t)[0]
        return effnet

    def update_weights_ema(self, src_model, beta=0.999):
        for self_params, src_params in zip(self.parameters(), src_model.parameters()):
            self_params.data = self_params.data * beta + src_params.data * (1 - beta)


class Prior(nn.Module):
    def __init__(self, c_in=16, c=1280, c_cond=1024, c_r=64, depth=16, nhead=16, latent_size=(12, 12), dropout=0.1):
        super().__init__()
        self.c_r = c_r
        self.projection = nn.Conv2d(c_in, c, kernel_size=1)
        self.cond_mapper = nn.Sequential(
            nn.Linear(c_cond, c),
            nn.LeakyReLU(0.2),
            nn.Linear(c, c),
        )

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(ResBlock(c, dropout=dropout))
            self.blocks.append(TimestepBlock(c, c_r))
            self.blocks.append(AttnBlock(c, c, nhead, self_attn=True, dropout=dropout))
        self.out = nn.Sequential(
            LayerNorm2d(c, elementwise_affine=False, eps=1e-6),
            nn.Conv2d(c, c_in * 2, kernel_size=1),
        )

        self.apply(self._init_weights)  # General init
        nn.init.normal_(self.projection.weight, std=0.02)  # inputs
        nn.init.normal_(self.cond_mapper[0].weight, std=0.02)  # conditionings
        nn.init.normal_(self.cond_mapper[-1].weight, std=0.02)  # conditionings
        nn.init.constant_(self.out[1].weight, 0)  # outputs

        # blocks
        for block in self.blocks:
            if isinstance(block, ResBlock):
                block.channelwise[-1].weight.data *= np.sqrt(1 / depth)
            elif isinstance(block, TimestepBlock):
                nn.init.constant_(block.mapper.weight, 0)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb

    def forward(self, x, r, c):
        x_in = x
        x = self.projection(x)
        c_embed = self.cond_mapper(c)
        r_embed = self.gen_r_embedding(r)
        for block in self.blocks:
            if isinstance(block, AttnBlock):
                x = block(x, c_embed)
            elif isinstance(block, TimestepBlock):
                x = block(x, r_embed)
            else:
                x = block(x)
        a, b = self.out(x).chunk(2, dim=1)
        # denoised = a / (1-(1-b).pow(2)).sqrt()
        return (x_in - a) / ((1 - b).abs() + 1e-5)

    def update_weights_ema(self, src_model, beta=0.999):
        for self_params, src_params in zip(self.parameters(), src_model.parameters()):
            self_params.data = self_params.data * beta + src_params.data * (1 - beta)


