from __future__ import annotations

import math
from math import log2
from typing import Callable
from pathlib import Path
from random import random
from shutil import rmtree
from collections import namedtuple

from PIL import Image

import torch
from torch import nn, arange, tensor, cat, stack
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as T
from torchvision.utils import save_image

from einops import rearrange, repeat, einsum, pack, unpack
from einops.layers.torch import Rearrange, Reduce

from accelerate import Accelerator
from ema_pytorch import EMA

from x_mlps_pytorch.ensemble import Ensemble
from x_transformers.x_transformers import Attention, RMSNorm

# constants

GuidedSamplerOutput = namedtuple('GuidedSamplerOutput', ('output', 'codes', 'commit_loss'))

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def sample_prob(prob):
    return random() < prob

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def l2dist(x1, x2, eps = 1e-12):
    return (x1 - x2).pow(2).sum(dim = -1).clamp(min = eps).sqrt()

def cdist(x1, x2, eps = 1e-12):
    is_mps = x1.device.type == 'mps'

    if not is_mps:
        return torch.cdist(x1, x2)

    dist = l2dist(x1, x2, eps = eps)
    dist = rearrange(dist, 'b k -> b 1 k')
    return dist

def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        unpacked, = unpack(out, ps, inv_pattern)
        return unpacked

    return packed, inverse

def Sequential(*mods):
    return nn.Sequential(*[*filter(exists, mods)])

# norms

class ChanRMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * (self.gamma + 1.) * self.scale

# gan losses

def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def hinge_gen_loss(fake):
    return -fake.mean()

def gradient_penalty(images, output, weight = 10, center = 0.):

    gradients = torch.autograd.grad(
        outputs = output,
        inputs = images,
        grad_outputs = torch.ones_like(output),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * (gradients.norm(2, dim = 1) - center).pow(2).mean()

def grad_layer_wrt_loss(loss, layer):
    grad = torch.autograd.grad(
        outputs = loss,
        inputs = layer,
        grad_outputs = torch.ones_like(loss),
        retain_graph = True,
        allow_unused = True
    )[0]
    return grad.detach() if exists(grad) else torch.zeros_like(layer)

def safe_div(numer, denom, eps = 1e-8):
    return numer / (denom + eps)

# classes

def split_and_prune_(network: Module):
    # given some parent network, calls split and prune for all guided samplers

    for m in network.modules():
        if isinstance(m, GuidedSampler):
            m.split_and_prune_()

class GuidedSampler(Module):
    def __init__(
        self,
        dim,                            # input feature dimension
        dim_query = 3,                  # channels of image (default 3 for rgb)
        codebook_size = 10,             # K in paper
        network: Module | None = None,
        distance_fn: Callable | None = None,
        chain_dropout_prob = 0.05,
        split_thres = 2.,
        prune_thres = 0.5,
        pre_network_activation: Module | None = None,
        post_network_activation: Module | None = None,
        prenorm = False,
        norm_module: Module | None = None,
        min_total_count_before_split_prune = 100,
        crossover_top2_prob = 0.,
        straight_through_distance_logits = False,
        stochastic = False,
        gumbel_noise_scale = 1.,
        patch_size = None,              # facilitate the future work where the guided sampler is done on patches
        separate_values = False,        # taking attention perspective, this will have the network also produce the values separately from the keys, using the dim_value below - (from this perspective, what the author has is actually a shared key/value hard attention)
        dim_values = None,              # default to dim_query (which is actually the key dimension)
    ):
        super().__init__()

        # normalization

        if prenorm and not exists(norm_module):
            norm_module = ChanRMSNorm(dim)

        self.norm = default(norm_module, nn.Identity())

        network_dim_out = dim_query

        # whether to have separate values passed on - can also pass both keys and values on

        self.dim_query = dim_query
        self.separate_values = separate_values

        if separate_values:
            dim_values = default(dim_values, dim_query)
            network_dim_out = network_dim_out + dim_values

        # the network / codebook

        if not exists(network):
            network = nn.Conv2d(dim, network_dim_out, 1, bias = False)

            if exists(post_network_activation) or exists(pre_network_activation):
                network = Sequential(pre_network_activation, network, post_network_activation)

        self.codebook_size = codebook_size
        self.to_key_values = Ensemble(network, ensemble_size = codebook_size)
        self.distance_fn = default(distance_fn, cdist)

        # chain dropout

        self.chain_dropout_prob = chain_dropout_prob

        # split and prune related

        self.register_buffer('counts', torch.zeros(codebook_size).long())

        self.split_thres = split_thres / codebook_size
        self.prune_thres = prune_thres / codebook_size
        self.min_total_count_before_split_prune = min_total_count_before_split_prune

        # improvisations

        self.crossover_top2_prob = crossover_top2_prob

        self.stochastic = stochastic
        self.gumbel_noise_scale = gumbel_noise_scale
        self.straight_through_distance_logits = straight_through_distance_logits

        # acting on patches instead of whole image, mentioned by author

        self.patch_size = patch_size
        self.acts_on_patches = exists(patch_size)

        if self.acts_on_patches:
            self.image_to_patches = Rearrange('b c (h p1) (w p2) -> b h w c p1 p2', p1 = patch_size, p2 = patch_size)
            self.patches_to_image = Rearrange('b h w c p1 p2 -> b c (h p1) (w p2)')

    @torch.no_grad()
    def split_and_prune_(
        self
    ):
        # following Algorithm 1 in the paper

        counts = self.counts
        total_count = counts.sum()

        if self.codebook_size < 2 or total_count < self.min_total_count_before_split_prune:
            return

        top2_values, top2_indices = counts.topk(2, dim = -1)

        count_max, count_max_index = top2_values[0], top2_indices[0]
        count_min, count_min_index = counts.min(dim = -1)

        if (
            ((count_max / total_count) <= self.split_thres) &
            ((count_min / total_count) >= self.prune_thres)
        ).all():
            return

        codebook_params = self.to_key_values.param_values
        half_count_max = count_max // 2

        # update the counts

        self.counts[count_max_index] = half_count_max
        self.counts[count_min_index] = half_count_max + 1 # adds 1 to k_new for some reason

        # whether to crossover top 2

        should_crossover = sample_prob(self.crossover_top2_prob)

        # update the params

        for codebook_param in codebook_params:

            split = codebook_param[count_max_index]

            # whether to crossover
            if should_crossover:
                second_index = top2_indices[1]
                second_split = codebook_param[second_index]
                split = (split + second_split) / 2. # naive average for now

            # prune by replacement
            codebook_param[count_min_index].copy_(split)

            # take care of grad if present
            if exists(codebook_param.grad):
                codebook_param.grad[count_min_index].zero_()

    def forward_for_codes(
        self,
        features,      # (b d h w)
        codes,         # (b) | ()
        residual = None
    ):
        batch = features.shape[0]

        features = self.norm(features)

        # handle patches

        if self.acts_on_patches:

            if codes.numel() == 1:
                codes = repeat(codes, ' -> b', b = features.shape[0])

            features = self.image_to_patches(features)
            b, h, w = features.shape[:3]
            features, inverse_pack = pack_one(features, '* c h w')

            if exists(residual):
                residual = self.image_to_patches(residual)
                residual, _ = pack_one(residual, '* c h w')

            if codes.ndim == 1:
                codes = repeat(codes, 'b -> (b h w)', h = h, w = w)
            elif codes.ndim == 3:
                codes = rearrange(codes, 'b h w -> (b h w)')
            else:
                codes = repeat(codes, '... -> (b h w)', b = b, h = h, w = w)

        # if one code, just forward the selected network for all features
        # else each batch is matched with the corresponding code

        if codes.numel() == 1:
            sel_key_values = self.to_key_values.forward_one(features, id = codes.item())
        else:
            sel_key_values =  self.to_key_values(features, ids = codes, each_batch_sample = True)

        if self.separate_values:
            sel_key_values = sel_key_values[:, self.dim_query:]

        # handle patches

        if self.acts_on_patches:
            sel_key_values = inverse_pack(sel_key_values)
            sel_key_values = self.patches_to_image(sel_key_values)

        if exists(residual):
            sel_key_values = sel_key_values + residual

        return sel_key_values

    def forward(
        self,
        features,       # (b d h w)
        query,          # (b c h w)
        return_distances = False,
        residual = None
    ):

        features = self.norm(features)

        # take care of maybe patching

        if self.acts_on_patches:
            features = self.image_to_patches(features)
            query = self.image_to_patches(query)

            features, _ = pack_one(features, '* c h w')
            query, inverse_pack = pack_one(query, '* c h w')

            if exists(residual):
                residual = self.image_to_patches(residual)
                residual, _ = pack_one(residual, '* c h w')

        # variables

        batch, device = query.shape[0], query.device

        key_values = self.to_key_values(features)

        # get the keys for distance

        keys = key_values[:, :, :self.dim_query] if self.separate_values else key_values
        keys_for_dist = keys + residual if exists(residual) else keys

        # get the l2 distance

        distance = self.distance_fn(
            rearrange(query, 'b ... -> b 1 (...)'),
            rearrange(keys_for_dist, 'k b ... -> b k (...)')
        )

        distance = rearrange(distance, 'b 1 k -> b k')

        logits = -distance

        # allow for a bit of stochasticity

        if self.stochastic and self.training:
            logits = logits + gumbel_noise(logits) * self.gumbel_noise_scale

        # select the code parameters that produced the image that is closest to the query

        if self.training and sample_prob(self.chain_dropout_prob):
            # handle the chain dropout

            codes = torch.randint(0, self.codebook_size, (batch,), device = device)

        else:
            codes = logits.argmax(dim = -1)

            if self.training:
                self.counts.scatter_add_(0, codes, torch.ones_like(codes))

        # some tensor gymnastics to select out the image across batch

        if not self.straight_through_distance_logits or not self.training:
            key_values = rearrange(key_values, 'k b ... -> b k ...')

            codes_for_indexing = rearrange(codes, 'b -> b 1')
            batch_for_indexing = arange(batch, device = device)[:, None]

            sel_key_values = key_values[batch_for_indexing, codes_for_indexing]
            sel_key_values = rearrange(sel_key_values, 'b 1 ... -> b ...')
        else:
            # variant treating the distance as attention logits

            attn = logits.softmax(dim = -1)
            one_hot = F.one_hot(codes, num_classes = self.codebook_size)

            st_one_hot = one_hot + attn - attn.detach()
            sel_key_values = einsum(key_values, st_one_hot, 'k b ..., b k -> b ...')

        # separate values logic

        if self.separate_values:
            sel_keys, sel_values = sel_key_values[:, :self.dim_query], sel_key_values[:, self.dim_query:]
        else:
            sel_keys, sel_values = sel_key_values, sel_key_values

        if exists(residual):
            sel_keys = sel_keys + residual
            sel_values = sel_values + residual

        # commit loss

        commit_loss = F.mse_loss(sel_keys, query)

        # maybe reconstitute patch dimensions

        if self.acts_on_patches:
            sel_values = inverse_pack(sel_values, '* c p1 p2')
            sel_values = self.patches_to_image(sel_values)

            codes = inverse_pack(codes, '*')

        # return the chosen feature, the code indices, and commit loss

        output = GuidedSamplerOutput(sel_values, codes, commit_loss)

        if not return_distances:
            return output

        return output, distance

# ddn

class Conv2dCroppedResidual(Module):
    # used in alphagenome

    def __init__(
        self,
        dim,
        dim_out,
        kernel_size,
        **kwargs
    ):
        super().__init__()
        assert dim >= dim_out
        self.pad = dim - dim_out
        self.conv = Block(dim, dim_out, 1)

    def forward(self, x):
        residual, length = x, x.shape[1]
        return self.conv(x) + residual[:, :(length - self.pad)]

class SqueezeExcite(Module):
    def __init__(
        self,
        dim,
        squeeze_factor = 4.
    ):
        super().__init__()
        dim_squeezed = int(max(32, dim // squeeze_factor))

        self.squeeze = Sequential(
            Reduce('b c h w -> b c 1 1', 'mean'),
            nn.Conv2d(dim, dim_squeezed, 1),
            nn.ReLU(),
            nn.Conv2d(dim_squeezed, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.squeeze(x)

class Block(Module):
    def __init__(
        self,
        dim,
        dim_out,
        kernel_size = 3,
        dropout = 0.
    ):
        super().__init__()
        self.norm = ChanRMSNorm(dim)
        self.act = nn.SiLU()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size, padding = kernel_size // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.contiguous()
        x = self.norm(x)
        x = self.act(x)
        x = self.proj(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        dropout = 0.
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.squeeze_excite = SqueezeExcite(dim_out)

        self.layerscale = nn.Parameter(torch.randn(dim_out, 1, 1) * 1e-6)

    def forward(self, x):
        res = x
        h = self.block1(x)
        h = self.block2(h)
        h = self.squeeze_excite(h)
        return h * self.layerscale + res

# discriminator

class DiscriminatorAttention(Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.attn = Attention(dim = dim, dim_head = 64, heads = 8)

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x_norm = self.norm(x)
        out = self.attn(x_norm)
        out = rearrange(out, 'b (h w) c -> b c h w', h = h, w = w)
        return out

class DiscriminatorBlock(Module):
    def __init__(
        self,
        input_channels,
        filters,
        downsample = True
    ):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
            nn.Conv2d(filters * 4, filters, 1)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)

        if exists(self.downsample):
            x = self.downsample(x)

        x = (x + res) * (1 / math.sqrt(2))
        return x

class Discriminator(Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        channels = 3,
        attn_res_layers = (16,),
        max_dim = 512
    ):
        super().__init__()
        image_size = (image_size, image_size) if not isinstance(image_size, tuple) else image_size
        min_image_resolution = min(image_size)

        num_layers = int(math.log2(min_image_resolution) - 2)
        attn_res_layers = attn_res_layers if isinstance(attn_res_layers, tuple) else (attn_res_layers,) * num_layers

        blocks = []

        layer_dims = [channels] + [(dim * 4) * (2 ** i) for i in range(num_layers + 1)]
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))

        blocks = []
        attn_blocks = []

        image_resolution = min_image_resolution

        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(layer_dims_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

            attn_block = None
            if image_resolution in attn_res_layers:
                attn_block = DiscriminatorAttention(dim = out_chan)

            attn_blocks.append(attn_block)

            image_resolution //= 2

        self.blocks = ModuleList(blocks)
        self.attn_blocks = ModuleList(attn_blocks)

        dim_last = layer_dims[-1]

        downsample_factor = 2 ** num_layers
        last_fmap_size = tuple(map(lambda n: n // downsample_factor, image_size))

        latent_dim = last_fmap_size[0] * last_fmap_size[1] * dim_last

        self.to_logits = Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding = 1),
            leaky_relu(),
            Rearrange('b ... -> b (...)'),
            nn.Linear(latent_dim, 1),
            Rearrange('b 1 -> b')
        )

    def forward(self, x):
        for block, attn_block in zip(self.blocks, self.attn_blocks):
            x = block(x)

            if exists(attn_block):
                x = attn_block(x) + x

        return self.to_logits(x)

class DDN(Module):
    def __init__(
        self,
        dim,
        dim_max = 1024,
        image_size = 256,
        channels = 3,
        codebook_size = 10,
        dropout = 0.,
        num_resnet_blocks = 2,
        guided_sampler_kwargs: dict = dict(),
        use_adversarial_loss = False,
        adversarial_loss_weight = 1.0,
        discr_base_dim = 16,
        discr_attn_res_layers = (16,),
    ):
        super().__init__()
        assert log2(image_size).is_integer()

        self.input_image_shape = (channels, image_size, image_size)

        # number of stages from 2x2 features

        stages = int(log2(image_size))

        self.num_stages = stages
        self.codebook_size = codebook_size

        # dimensions

        dim_mults = reversed([2 ** stage for stage in range(stages)])

        dims = [min(dim_max, dim * dim_mult) for dim_mult in dim_mults]

        dim_first = dims[0]

        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        # initial 2x2 features

        self.init_features = nn.Parameter(torch.randn(dim_first, 2, 2) * 1e-2)

        # layers

        self.layers = ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(dim_pairs):

            has_prev_sampler_output = ind != 0

            prev_sampled_dim = channels if has_prev_sampler_output else 0

            dim_in_with_maybe_prev = dim_in + prev_sampled_dim

            upsampler = nn.Sequential(
                nn.Upsample(scale_factor = 2, mode = 'bilinear'),
                Conv2dCroppedResidual(dim_in_with_maybe_prev, dim_out, 1)
            )

            resnet_block = nn.Sequential(*[
                ResnetBlock(dim_out, dropout = dropout) for _ in range(num_resnet_blocks)
            ])

            guided_sampler = GuidedSampler(
                dim = dim_out,
                dim_query = channels,
                codebook_size = codebook_size,
                **guided_sampler_kwargs
            )

            self.layers.append(ModuleList([
                upsampler,
                resnet_block,
                guided_sampler
            ]))

        # discriminator

        self.use_adversarial_loss = use_adversarial_loss
        self.adversarial_loss_weight = adversarial_loss_weight
        self.discr = None

        if use_adversarial_loss:
            self.discr = Discriminator(
                image_size = image_size,
                dim = discr_base_dim,
                channels = channels,
                attn_res_layers = discr_attn_res_layers
            )


    def guided_sampler_codes_param_names(self):

        names = []

        for name, _ in self.named_parameters():
            sub_names = set(name.split('.'))

            if 'to_key_values' not in sub_names:
                continue

            names.append(name)

        return set(names)

    def split_and_prune_(self):
        split_and_prune_(self)

    @property
    def device(self):
        return next(self.parameters()).device

    def sample(
        self,
        batch_size = None,
        codes = None,  # (b stages)
        return_codes = False
    ):
        was_training = self.training
        self.eval()

        assert exists(batch_size) ^ exists(codes)

        batch_size = default(batch_size, codes.shape[0] if exists(codes) else None)

        # if only batch size sent in, random codes

        if not exists(codes):
            codes = torch.randint(0, self.codebook_size, (batch_size, self.num_stages), device = self.device)

        # init features

        features = repeat(self.init_features, '... -> b ...', b = batch_size)

        # sampled output of a stage

        sampled_output = None
        rgb_residual = None

        for (upsampler, resnet_block, guided_sampler), layer_codes in zip(self.layers, codes.unbind(dim = -1)):

            if exists(sampled_output):
                features = cat((sampled_output, features), dim = 1)

            features = upsampler(features)

            features = resnet_block(features)

            if exists(rgb_residual):
                height, width = features.shape[-2:]
                rgb_residual = F.interpolate(rgb_residual, (height, width), mode = 'bilinear')

            sampled_output = guided_sampler.forward_for_codes(features, layer_codes, residual = rgb_residual)

            rgb_residual = sampled_output

        self.train(was_training)

        # last sampled output

        if not return_codes:
            return sampled_output

        return sampled_output, codes

    def forward(
        self,
        images,
        return_intermediates = False,
        return_discr_loss = False,
        apply_grad_penalty = True
    ):
        assert images.shape[1:] == self.input_image_shape
        batch = images.shape[0]

        # init features

        features = repeat(self.init_features, '... -> b ...', b = batch)

        losses = []
        codes = []
        sampled_outputs = []

        rgb_residual = None

        for upsampler, resnet_block, guided_sampler in self.layers:

            if len(sampled_outputs) > 0:
                features = cat((sampled_outputs[-1], features), dim = 1)

            features = upsampler(features)

            features = resnet_block(features)

            # query image for guiding is just input images resized

            height, width = features.shape[-2:]
            query_images = F.interpolate(images, (height, width), mode = 'bilinear')

            # handle rgb residual

            if exists(rgb_residual):
                rgb_residual = F.interpolate(rgb_residual, (height, width), mode = 'bilinear')

            # guided sampler

            sampled_output, layer_code, layer_loss = guided_sampler(features, query_images, residual = rgb_residual)

            rgb_residual = sampled_output

            # losses, codes, outputs

            sampled_outputs.append(sampled_output)
            codes.append(layer_code)
            losses.append(layer_loss)

        # losses summed across layers

        total_loss = sum(losses)

        # discriminator loss if required

        if return_discr_loss:
            assert exists(self.discr), 'discriminator must exist to train it'
            recon_images = sampled_outputs[-1].detach()

            # requires grad for gradient penalty
            images.requires_grad_()

            recon_discr_logits, real_discr_logits = map(self.discr, (recon_images, images))
            discr_loss = hinge_discr_loss(recon_discr_logits, real_discr_logits)

            if apply_grad_penalty:
                gp = gradient_penalty(images, real_discr_logits)
                discr_loss = discr_loss + gp

            return discr_loss

        # generator loss

        if self.use_adversarial_loss:
            recon_images = sampled_outputs[-1]
            gen_loss = hinge_gen_loss(self.discr(recon_images))

            # calculate adaptive weight
            last_guided_sampler = self.layers[-1][-1]
            last_dec_layer = next(last_guided_sampler.parameters())

            if exists(last_dec_layer):
                norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p = 2)
                norm_grad_wrt_recon_loss = grad_layer_wrt_loss(total_loss, last_dec_layer).norm(p = 2)

                adaptive_weight = safe_div(norm_grad_wrt_recon_loss, norm_grad_wrt_gen_loss)
                adaptive_weight.clamp_(max = 1e4)
            else:
                adaptive_weight = 1.0

            total_loss = total_loss + (adaptive_weight * self.adversarial_loss_weight * gen_loss)

        if not return_intermediates:
            return total_loss

        codes = stack(codes, dim = -1)

        return total_loss, (codes, sampled_outputs)

# dataset & trainer

class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ('jpg', 'jpeg', 'png', 'tiff', 'webp')
    ):
        super().__init__()
        self.folder = Path(folder)
        self.image_size = image_size
        self.paths = [p for ext in exts for p in self.folder.glob(f'**/*.{ext}')]
        assert len(self.paths) > 0, f'no images found in {folder}'

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        return self.transform(img)

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

class Trainer(Module):
    def __init__(
        self,
        ddn: dict | DDN,
        *,
        dataset: dict | Dataset,
        num_train_steps = 70_000,
        learning_rate = 3e-4,
        weight_decay = 1e-3,
        batch_size = 32,
        grad_accum_every = 1,
        checkpoints_folder: str = './checkpoints',
        results_folder: str = './results',
        save_results_every: int = 100,
        checkpoint_every: int = 1000,
        num_samples: int = 16,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        use_ema = True,
        max_grad_norm = 0.5,
        apply_grad_penalty_every = 4
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        if isinstance(dataset, dict):
            dataset = ImageDataset(**dataset)

        if isinstance(ddn, dict):
            ddn = DDN(**ddn)

        self.model = ddn

        self.apply_grad_penalty_every = apply_grad_penalty_every

        self.use_ema = use_ema
        self.ema_model = None

        if self.is_main and use_ema:
            self.ema_model = EMA(
                self.model,
                forward_method_names = ('sample',),
                param_or_buffer_names_no_ema = ddn.guided_sampler_codes_param_names(),
                **ema_kwargs
            )

            self.ema_model.to(self.accelerator.device)

        # optimizer, dataloader, and all that

        all_parameters = set(self.model.parameters())
        discr_parameters = set(self.model.discr.parameters()) if exists(self.model.discr) else set()
        vae_parameters = all_parameters - discr_parameters

        self.optimizer = AdamW(vae_parameters, lr = learning_rate, weight_decay = weight_decay, **adam_kwargs)
        if exists(self.model.discr):
            self.discr_optimizer = AdamW(discr_parameters, lr = learning_rate, weight_decay = weight_decay, **adam_kwargs)
        else:
            self.discr_optimizer = None

        self.dl = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)

        if exists(self.discr_optimizer):
            self.model, self.optimizer, self.discr_optimizer, self.dl = self.accelerator.prepare(self.model, self.optimizer, self.discr_optimizer, self.dl)
        else:
            self.model, self.optimizer, self.dl = self.accelerator.prepare(self.model, self.optimizer, self.dl)

        self.num_train_steps = num_train_steps

        # folders

        self.checkpoints_folder = Path(checkpoints_folder)
        self.results_folder = Path(results_folder)

        if self.results_folder.exists() and self.is_main:
            rmtree(str(self.results_folder))

        self.checkpoints_folder.mkdir(exist_ok = True, parents = True)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.checkpoint_every = checkpoint_every
        self.save_results_every = save_results_every

        self.num_sample_rows = int(math.sqrt(num_samples))
        assert (self.num_sample_rows ** 2) == num_samples, f'{num_samples} must be a square'
        self.num_samples = num_samples

        assert self.checkpoints_folder.is_dir()
        assert self.results_folder.is_dir()

        self.max_grad_norm = max_grad_norm
        self.grad_accum_every = grad_accum_every

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    def save(self, path):
        if not self.is_main:
            return

        save_package = dict(
            model = self.unwrapped_model.state_dict(),
            optimizer = self.optimizer.state_dict(),
        )

        if exists(self.discr_optimizer):
            save_package['discr_optimizer'] = self.discr_optimizer.state_dict()

        if exists(self.ema_model):
            save_package['ema_model'] = self.ema_model.state_dict()

        torch.save(save_package, str(self.checkpoints_folder / path))

    def load(self, path):
        load_package = torch.load(path, map_location = self.accelerator.device)

        self.unwrapped_model.load_state_dict(load_package["model"])

        if 'ema_model' in load_package and exists(self.ema_model):
            self.ema_model.load_state_dict(load_package["ema_model"])

        self.optimizer.load_state_dict(load_package["optimizer"])

        if 'discr_optimizer' in load_package and exists(self.discr_optimizer):
            self.discr_optimizer.load_state_dict(load_package["discr_optimizer"])

    def log(self, *args, **kwargs):
        return self.accelerator.log(*args, **kwargs)

    def log_images(self, images, **kwargs):
        return self.log({'samples': images}, **kwargs)

    @torch.no_grad()
    def sample(self, fname):
        eval_model = default(self.ema_model, self.model)

        sampled = eval_model.sample(batch_size = self.num_samples)

        sampled = rearrange(sampled, '(row col) c h w -> c (row h) (col w)', row = self.num_sample_rows)
        sampled.clamp_(0., 1.)

        save_image(sampled, fname)
        return sampled

    def forward(self):

        dl = cycle(self.dl)

        for ind in range(self.num_train_steps):
            step = ind + 1

            self.model.train()

            self.optimizer.zero_grad()
            total_loss = 0.

            for _ in range(self.grad_accum_every):
                data = next(dl)
                loss = self.model(data)

                self.accelerator.backward(loss / self.grad_accum_every)
                total_loss += loss.item() / self.grad_accum_every

            logs = dict(loss = total_loss)

            print_str = f'[{step}] loss: {total_loss:.3f}'

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            # discriminator update

            if exists(self.unwrapped_model.discr):
                self.discr_optimizer.zero_grad()
                total_discr_loss = 0.
                apply_grad_penalty = divisible_by(step, self.apply_grad_penalty_every)

                for _ in range(self.grad_accum_every):
                    data = next(dl)
                    discr_loss = self.model(data, return_discr_loss = True, apply_grad_penalty = apply_grad_penalty)

                    self.accelerator.backward(discr_loss / self.grad_accum_every)
                    total_discr_loss += discr_loss.item() / self.grad_accum_every

                if exists(self.max_grad_norm):
                    self.accelerator.clip_grad_norm_(self.unwrapped_model.discr.parameters(), self.max_grad_norm)

                self.discr_optimizer.step()

                logs['discr_loss'] = total_discr_loss
                print_str += f' | discr loss: {total_discr_loss:.3f}'

            self.accelerator.print(print_str, flush = True)

            self.log(logs, step = step)

            self.unwrapped_model.split_and_prune_() # call split and prune after update

            if self.is_main and self.use_ema:
                self.ema_model.update()

            self.accelerator.wait_for_everyone()

            if self.is_main:

                if divisible_by(step, self.save_results_every):

                    sampled = self.sample(fname = str(self.results_folder / f'results.{step}.png'))

                    self.log_images(sampled, step = step)

                if divisible_by(step, self.checkpoint_every):
                    self.save(f'checkpoint.{step}.pt')

            self.accelerator.wait_for_everyone()

        print('training complete')
