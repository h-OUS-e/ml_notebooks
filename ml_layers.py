"""
Transformer building blocks extracted from `learning_transformers.ipynb`.

The notebook builds the same ideas up in stages and re-defines names as it goes
(`MultiheadAttention` -> `FlashMultiheadAttention` -> `MHA_RoPE`, `AttnBlock` x3,
`ViT_Encoder` x2). Here every stage keeps its own name so nothing shadows anything:

Patching
    patchify / unpatchify

Attention (in the order the notebook builds them)
    MultiheadAttention       naive MHA, explicit score matrix + softmax
    FlashMultiheadAttention  same math via F.scaled_dot_product_attention (SDPA)
    GQA                      grouped-query attention (n_kv_heads=1 -> MQA)
    MHA_RoPE                 SDPA + rotary embeddings (1d or 2d) + optional causal mask

RoPE helpers
    rotate_pairs, build_rope_1d_cache, apply_rope_1d,
    build_rope_2d_cache, apply_rope_2d

Blocks
    AttnBlockSimple          pre-LN block over the naive MultiheadAttention
    AttnBlock                pre-LN block over MHA_RoPE (rope + causal capable)

Models
    ViT_Encoder / ViT_Decoder / ViT_AutoEncoder            MSE reconstruction
    ViT_EncoderCLS                                        cls vs mean vs max pooling
    ViT_DecoderGaussian / ViT_AutoEncoderGaussian         Gaussian NLL head
    ViT_DecoderCategorical / ViT_AutoEncoderCategorical   binned cross-entropy head
    Classifier                                            linear probe on frozen latents
    AutoRegressivePredictor                               causal next-step predictor

One behaviour change vs. the notebook: there, blocks were built with
`MHA_RoPE(dim, n_heads, use_rope=not use_pos)`. `False is not None` is True, so RoPE
was applied even in the "learned positional encoding" runs. Here `use_pos` (learned
embedding) and `use_rope` (None | "1d" | "2d") are independent, so `use_pos=True,
use_rope=None` gives a genuinely rope-free model.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import einops

__all__ = [
    "patchify",
    "unpatchify",
    "MultiheadAttention",
    "FlashMultiheadAttention",
    "GQA",
    "rotate_pairs",
    "build_rope_1d_cache",
    "apply_rope_1d",
    "build_rope_2d_cache",
    "apply_rope_2d",
    "MHA_RoPE",
    "AttnBlockSimple",
    "AttnBlock",
    "ViT_Encoder",
    "ViT_Decoder",
    "ViT_AutoEncoder",
    "ViT_EncoderCLS",
    "ViT_DecoderGaussian",
    "ViT_AutoEncoderGaussian",
    "ViT_DecoderCategorical",
    "ViT_AutoEncoderCategorical",
    "Classifier",
    "AutoRegressivePredictor",
]


# --------------------------------------------------------------------------------------
# E1. Patchify / un-patchify
# --------------------------------------------------------------------------------------

def patchify(images: torch.Tensor, patch_size: int):
    """
    Patchifies images from (B, 1, 28, 28) to (B, 16, 49)

    Args:
        images: (B, C, H, W)
    """
    # Extract patches
    patches = F.unfold(images, kernel_size=patch_size, padding=0, stride=patch_size)
    # P are pixels inside a patch. G are patch grid position
    patches = einops.rearrange(patches, "... P G -> ... G P")
    return patches


def unpatchify(patches: torch.Tensor, patch_size: int, grid_size: int):
    """
    Unpatchify patches to the original image from (B, 16, 49) to (B, 1, 28, 28)

    Args:
        patches: (B, 16, 49)
    """
    # Reshape patches. ph pw are pixels inside a patch. gh gw are patch grid position
    images = einops.rearrange(
        patches, "... (gh gw) (ph pw) -> ... 1 (gh ph) (gw pw)", gh=grid_size, ph=patch_size
    )
    return images


# --------------------------------------------------------------------------------------
# E2.a. Multi-head self-attention (naive, written out by hand)
# --------------------------------------------------------------------------------------

class MultiheadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.head_dim = dim // n_heads
        self.n_heads = n_heads
        self.proj = nn.Linear(dim, 3 * self.head_dim * n_heads, bias=False)

        # The last layer that fuses multiple attention heads into a single layer
        self.to_out = nn.Linear(self.head_dim * n_heads, dim, bias=False)

    def forward(self, seq):
        """
        Args:
            seq (B, N, dim)

        Returns:
            out: (B, N, dim)
            attention: (B, n_heads, N, N)
        """
        # project to Q, K and V
        qkv = self.proj(seq)
        # Rearrange to move head outside to perform attention calculation PER head
        qkv = einops.rearrange(qkv, "b n (h d) -> b h n d", h=self.n_heads)
        # Divide d into 3 chunks d/3 for each query, key and value
        q, k, v = qkv.chunk(3, dim=-1)

        # Dot product of Queries and Keys.
        # Contracted dimension must share the same var name in einops (here it is 'd')
        score_matrix = einops.einsum(q, k, "... n1 d, ... n2 d -> ... n1 n2")

        # scale by head size
        score_matrix = score_matrix / (self.head_dim ** 0.5)

        # Applying softmax to the score matrix
        attention = torch.softmax(score_matrix, dim=-1)

        # Weighting attention by the value
        w = attention @ v

        # Move heads back to dim
        out = einops.rearrange(w, "b h n d -> b n (h d)")

        # Fuse multi-heads into a linear layer
        out = self.to_out(out)

        return out, attention


# --------------------------------------------------------------------------------------
# E7. Same attention, but through torch's fused SDPA kernel (flash attention)
# --------------------------------------------------------------------------------------

class FlashMultiheadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.head_dim = dim // n_heads
        self.n_heads = n_heads
        self.proj = nn.Linear(dim, 3 * self.head_dim * n_heads, bias=False)

        # The last layer that fuses multiple attention heads into a single layer
        self.to_out = nn.Linear(self.head_dim * n_heads, dim, bias=False)

    def forward(self, seq, get_attn_matrix: bool = False):
        """
        Args:
            seq (B, N, dim)

        Returns:
            out: (B, N, dim)
            attention: (B, n_heads, N, N) or None when get_attn_matrix is False
        """
        # project to Q, K and V
        qkv = self.proj(seq)
        # Rearrange to move head outside to perform attention calculation PER head
        qkv = einops.rearrange(qkv, "b n (h d) -> b h n d", h=self.n_heads)
        # Divide d into 3 chunks d/3 for each query, key and value
        q, k, v = qkv.chunk(3, dim=-1)

        attention = None
        if get_attn_matrix:
            # SDPA never materializes the score matrix, so rebuild it only when asked
            score_matrix = einops.einsum(q, k, "... n1 d, ... n2 d -> ... n1 n2")
            score_matrix = score_matrix / (self.head_dim ** 0.5)
            attention = torch.softmax(score_matrix, dim=-1)

        # Torch's flash attention formula, same result just faster. Also known as "SDPA"
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # (B, n_heads, N, head_dim)

        # Move heads back to dim
        out = einops.rearrange(out, "b h n d -> b n (h d)")

        # Fuse multi-heads into a linear layer
        out = self.to_out(out)

        return out, attention


# --------------------------------------------------------------------------------------
# MHA vs MQA vs GQA. n_kv_heads == n_heads -> MHA, n_kv_heads == 1 -> MQA.
# Mainly shrinks the KV cache; it does not really buy compute or memory at train time.
# --------------------------------------------------------------------------------------

class GQA(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.head_dim = dim // n_heads
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0, \
            "num of heads must be divisible by num of kv heads to repeat_interleave"

        self.proj_to_q = nn.Linear(dim, n_heads * self.head_dim)       # N heads for Q (Query)
        self.proj_to_kv = nn.Linear(dim, 2 * n_kv_heads * self.head_dim)  # fewer heads for K and V

        # The last layer that fuses multiple attention heads into a single layer
        self.to_out = nn.Linear(self.head_dim * n_heads, dim, bias=False)

    def forward(self, seq, get_attn_matrix: bool = False):
        """
        Args:
            seq (B, N, dim)

        Returns:
            out: (B, N, dim)
            attention: (B, n_heads, N, N) or None when get_attn_matrix is False
        """
        # project to Q, K and V
        q = self.proj_to_q(seq)
        kv = self.proj_to_kv(seq)

        # Rearrange to move head outside to perform attention calculation PER head
        q = einops.rearrange(q, "b n (num_heads head_dim) -> b num_heads n head_dim",
                             num_heads=self.n_heads)
        kv = einops.rearrange(kv, "b n (num_kv_heads head_dim) -> b num_kv_heads n head_dim",
                              num_kv_heads=self.n_kv_heads)

        # Split the kv projection into keys and values
        k, v = kv.chunk(2, dim=-1)

        # Expand k and v to match q for matmul. so heads [A, B] -> [A, A, A, A, B, B, B, B].
        # Interleaving by hand beat SDPA's enable_gqa=True in the notebook benchmarks.
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        attention = None
        if get_attn_matrix:
            score_matrix = einops.einsum(q, k, "... n1 d, ... n2 d -> ... n1 n2")
            score_matrix = score_matrix / (self.head_dim ** 0.5)
            attention = torch.softmax(score_matrix, dim=-1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # (B, n_heads, N, head_dim)

        # Move heads back to dim
        out = einops.rearrange(out, "b h n d -> b n (h d)")

        # Fuse multi-heads into a linear layer
        out = self.to_out(out)

        return out, attention


# --------------------------------------------------------------------------------------
# RoPE: rotate q and k by an angle proportional to their position, so that after the
# rotation q_i . k_j depends only on (i - j), the relative position.
# --------------------------------------------------------------------------------------

def rotate_pairs(x, angles):
    """
    Args:
        x: (B, n_heads, seq_len (N), head_dim)
        angles: (N_cache, head_dim/2)
    """
    N = x.shape[-2]
    # [:N] ensures our angles are clipped up to the sequence length of the data input
    cos = angles.cos()[:N][None, None, :, :]
    sin = angles.sin()[:N][None, None, :, :]

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # Applying rotation parts in euclidean space
    x_rotated = torch.empty_like(x)
    x_rotated[..., 0::2] = x_even * cos - x_odd * sin
    x_rotated[..., 1::2] = x_even * sin + x_odd * cos

    return x_rotated


def build_rope_1d_cache(N, head_dim, device, base=10000):
    assert head_dim % 2 == 0, "head dimension must be even to apply rotation to it"

    positions = torch.arange(N, device=device)
    inverse_frequences = base ** (-torch.arange(0, head_dim, 2, device=device).float() / head_dim)

    # Getting angles (shape: (N, head_dim/2))
    angles = einops.einsum(positions.float(), inverse_frequences, "n, d -> n d")

    return angles


def apply_rope_1d(x, angles):
    """
    Rotates pairs of dims by a 1d position.

    Args:
        x: (B, n_heads, seq_len (N), head_dim)
        angles: (N, head_dim / 2)
    """
    return rotate_pairs(x, angles)


def build_rope_2d_cache(grid_size, head_dim, device, base=10000):
    """
    NOTE: this only works for square grids for now since we assume one grid size.
    """
    assert head_dim % 4 == 0, "head dimension must be divisible by 4 to apply 2D rotation to it"
    half_dim = head_dim // 2

    pos_y, pos_x = torch.meshgrid(
        torch.arange(grid_size, device=device),
        torch.arange(grid_size, device=device),
        indexing="ij",
    )

    pos_y = pos_y.reshape(-1)
    pos_x = pos_x.reshape(-1)

    inverse_frequences = base ** (-torch.arange(0, half_dim, 2, device=device).float() / half_dim)

    # Getting angles (shape: (N, head_dim/4) each)
    angles_y = einops.einsum(pos_y.float(), inverse_frequences, "n, d -> n d")
    angles_x = einops.einsum(pos_x.float(), inverse_frequences, "n, d -> n d")

    return angles_y, angles_x


def apply_rope_2d(x, angles_yx):
    """
    Rotates the first half of head_dim by the row position and the second half by the
    column position, thus rope_2d.

    Args:
        x: (B, n_heads, seq_len (N), head_dim)
        angles_yx: tuple of 2 tensors, each (N, head_dim / 4)
    """
    angles_y, angles_x = angles_yx
    D_half = x.shape[-1] // 2

    x_y = x[..., :D_half]
    x_x = x[..., D_half:]

    x_rotated_y = rotate_pairs(x_y, angles_y)
    x_rotated_x = rotate_pairs(x_x, angles_x)

    return torch.cat([x_rotated_y, x_rotated_x], dim=-1)


class MHA_RoPE(nn.Module):
    """
    SDPA attention with rotary position embeddings and an optional causal mask.

    Args:
        use_rope: None (no rope), "1d" (sequence position), "2d" (square patch grid)
        is_causal: token t may only attend to tokens <= t
        n_kv_heads: None or n_heads means normal MHA. 1 means MQA. >1 means GQA
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int | None = None,
                 is_causal: bool = False, use_rope: str | None = "2d"):
        super().__init__()
        self.head_dim = dim // n_heads
        self.n_heads = n_heads
        self.use_rope = use_rope
        self.is_causal = is_causal
        self.last_attention = None
        
        # switching to GQA if n_kv_heads is not none
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = n_heads // self.n_kv_heads
        assert n_heads % self.n_kv_heads == 0, "num of heads must be divisible by num of kv heads to repeat_interleave"
        
        # Split projections: one fused qkv proj only works when n_kv_heads == n_heads
        self.proj_to_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.proj_to_kv = nn.Linear(dim, 2 * self.n_kv_heads * self.head_dim, bias=False)
        
        # The last layer that fuses multiple attention heads into a single layer
        self.to_out = nn.Linear(self.head_dim * n_heads, dim, bias=False)

        # Cache the rope angles as a buffer: non-trainable state that follows .to(device),
        # with persistent=False to keep this recomputable cache out of state_dict().
        if use_rope == "2d":
            self.register_buffer("rope_angles_y", None, persistent=False)
            self.register_buffer("rope_angles_x", None, persistent=False)
        elif use_rope is not None:
            self.register_buffer("rope_angles", None, persistent=False)


    def forward(self, seq, get_attn_matrix: bool = False):
        """
        Args:
            seq (B, N, dim)

        Returns:
            out: (B, N, dim). The attention matrix, when requested via get_attn_matrix,
            is stashed on self.last_attention instead of being returned.
        """
        B, N, dim = seq.size()

        # project to Q, K and V. Q gets n_heads, K/V get n_kv_heads
        q = einops.rearrange(self.proj_to_q(seq), "b n (h d) -> b h n d", h=self.n_heads)
        kv = einops.rearrange(self.proj_to_kv(seq), "b n (h d) -> b h n d", h=self.n_kv_heads)
        k, v = kv.chunk(2, dim=-1)

        if self.use_rope == "2d":
            # Build the rope angle matrix if not built yet, or if the sequence grew
            if (self.rope_angles_y is None) or (self.rope_angles_y.shape[0] < N):
                grid_size = int(N ** 0.5)
                assert grid_size * grid_size == N  # todo: non-square grid rope
                self.rope_angles_y, self.rope_angles_x = build_rope_2d_cache(
                    grid_size, self.head_dim, seq.device
                )

            q = apply_rope_2d(q, (self.rope_angles_y, self.rope_angles_x))
            k = apply_rope_2d(k, (self.rope_angles_y, self.rope_angles_x))

        elif self.use_rope is not None:
            # Build the rope angle matrix if not built yet, or if the sequence grew
            if (self.rope_angles is None) or (self.rope_angles.shape[0] < N):
                self.rope_angles = build_rope_1d_cache(N, self.head_dim, seq.device)

            q = apply_rope_1d(q, self.rope_angles)
            k = apply_rope_1d(k, self.rope_angles)

        # Expand kv to match q. Heads [A, B] -> [A, A, A, A, B, B, B, B]
        # No-op when n_kv_heads == n_heads. SDPA's enable_gqa=True does this internally
        # but benchmarked worse, hence the manual interleave.
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
            
        self.last_attention = None
        if get_attn_matrix:
            # SDPA never materializes the score matrix, so rebuild it only when asked
            score_matrix = einops.einsum(q, k, "... n1 d, ... n2 d -> ... n1 n2")
            score_matrix = score_matrix / (self.head_dim ** 0.5)

            if self.is_causal:
                causal_mask = torch.triu(
                    torch.ones(N, N, device=seq.device, dtype=torch.bool), diagonal=1
                )
                score_matrix = score_matrix.masked_fill(causal_mask, float("-inf"))

            attention = torch.softmax(score_matrix, dim=-1)
            self.last_attention = attention.detach()

        # Torch's flash attention formula, same result just faster. Also known as "SDPA"
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)

        # Move heads back to dim
        out = einops.rearrange(out, "b h n d -> b n (h d)")

        # Fuse multi-heads into a linear layer
        out = self.to_out(out)

        return out


# --------------------------------------------------------------------------------------
# E2.b. Pre-LN transformer blocks
# --------------------------------------------------------------------------------------

class AttnBlockSimple(nn.Module):
    """Pre-LN block over the hand-written MultiheadAttention (no rope, no mask)."""

    def __init__(self, dim: int, n_heads: int, attn_multiple: int = 4):
        super().__init__()
        self.attention = MultiheadAttention(dim, n_heads)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, attn_multiple * dim),
            nn.GELU(),
            nn.Linear(attn_multiple * dim, dim),
        )

    def forward(self, x):
        # attending to tokens across patches with a residual
        out, _ = self.attention(self.layer_norm1(x))
        x = x + out

        # channel mixing per token - a "scratch space" to make sense of the tokens we attended to
        out = self.ff(self.layer_norm2(x))
        x = x + out

        return x


class AttnBlock(nn.Module):
    """
    Pre-LN block over MHA_RoPE. Same block whether attention is bidirectional or causal,
    the only difference is is_causal on the attention.

    To read the attention matrix of a block:
        with torch.no_grad():
            h = model.encoder(x)
            block = model.attn_blocks[0]
            _ = block(h, get_attn_matrix=True)
            attention_matrix = block.attention.last_attention
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int | None = None,
                 is_causal: bool = False, use_rope: str | None = "2d",
                 attn_multiple: int = 4):
        super().__init__()
        self.attention = MHA_RoPE(dim, n_heads, n_kv_heads=n_kv_heads,
                                  is_causal=is_causal, use_rope=use_rope)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, attn_multiple * dim),
            nn.GELU(),
            nn.Linear(attn_multiple * dim, dim),
        )

    def forward(self, x, get_attn_matrix: bool = False):
        # attending to tokens across patches with a residual
        out = self.attention(self.layer_norm1(x), get_attn_matrix=get_attn_matrix)
        x = x + out

        # channel mixing per token - a "scratch space" to make sense of the tokens we attended to
        out = self.ff(self.layer_norm2(x))
        x = x + out

        return x


# --------------------------------------------------------------------------------------
# E2.c. ViT autoencoder
# --------------------------------------------------------------------------------------

class ViT_Encoder(nn.Module):
    def __init__(self, dim: int, n_heads: int, z_dim: int, P: int, N: int, k: int = 2,
                 use_pos: bool = False, use_rope: str | None = "2d"):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(P, dim), nn.GELU())

        # Stack attention blocks
        self.attn_blocks = nn.Sequential(
            *[AttnBlock(dim, n_heads, use_rope=use_rope) for _ in range(k)]
        )

        # Linear layer to encode to a lower dim (not sure if we should use GELU here or not)
        self.ff2 = nn.Sequential(nn.GELU(), nn.Linear(dim, z_dim))

        # Layer for positional encoding
        self.pos = nn.Parameter(torch.randn(1, N, dim) * 0.02)  # N = grid_size ** 2
        self.use_pos = use_pos

    def forward(self, seq):
        """
        Args:
            seq (B, N, P): P is patch_size**2 and N is number of patch rows x columns

        Returns:
            z (B, N, z_dim)
        """
        # embed sequence
        x = self.ff(seq)

        # positional encoding
        if self.use_pos:
            x = x + self.pos

        # Attention blocks
        x = self.attn_blocks(x)

        # project sequence to compressed latent vector
        z = self.ff2(x)
        return z


class ViT_Decoder(nn.Module):
    def __init__(self, dim: int, n_heads: int, z_dim: int, P: int, N: int, k: int = 2,
                 use_pos: bool = False, use_rope: str | None = "2d"):
        super().__init__()

        # Linear layer to decode from a lower dim to a higher one
        self.ff = nn.Sequential(nn.GELU(), nn.Linear(z_dim, dim))

        # Stack attention blocks
        self.attn_blocks = nn.Sequential(
            *[AttnBlock(dim, n_heads, use_rope=use_rope) for _ in range(k)]
        )

        # Linear layer to unembed tokens. It shouldn't end with an activation since we
        # want raw pixel values (basic ML stuff)
        self.ff2 = nn.Sequential(nn.GELU(), nn.Linear(dim, P))

        # Layer for positional encoding
        self.pos = nn.Parameter(torch.randn(1, N, dim) * 0.02)  # N = grid_size ** 2
        self.use_pos = use_pos

    def forward(self, seq):
        """
        Args:
            seq (B, N, z_dim): latent vector encoded via ViT_Encoder

        Returns:
            out (B, N, P): P is patch_size**2 and N is number of patch rows x columns
        """
        # embed sequence
        x = self.ff(seq)

        # positional encoding
        if self.use_pos:
            x = x + self.pos

        x = self.attn_blocks(x)

        # project sequence to uncompressed seq
        x = self.ff2(x)
        return x


class ViT_AutoEncoder(nn.Module):
    def __init__(self, dim: int, n_heads: int, z_dim: int, patch_size: int, grid_size: int,
                 k: int = 2, use_pos: bool = False, use_rope: str | None = "2d"):
        super().__init__()

        self.patch_size = patch_size
        self.grid_size = grid_size

        self.encoder = ViT_Encoder(dim, n_heads, z_dim, patch_size ** 2, grid_size ** 2,
                                   k=k, use_pos=use_pos, use_rope=use_rope)
        self.decoder = ViT_Decoder(dim, n_heads, z_dim, patch_size ** 2, grid_size ** 2,
                                   k=k, use_pos=use_pos, use_rope=use_rope)

    def forward(self, images):
        """
        Args:
            images (B, C, H, W)
        """
        image_patches = patchify(images, self.patch_size)
        z = self.encoder(image_patches)
        seq = self.decoder(z)
        recon = unpatchify(seq, self.patch_size, self.grid_size)

        return recon, z


# --------------------------------------------------------------------------------------
# E4. CLS token vs pooling
# --------------------------------------------------------------------------------------

class ViT_EncoderCLS(nn.Module):
    """
    Classifier encoder used to compare a learned cls token against mean/max pooling.

    z_dim is unused (kept so the notebook's positional call still works) - the head
    projects straight to num_classes.
    """

    def __init__(self, dim: int, n_heads: int, z_dim: int, P: int, N: int, k: int = 2,
                 use_pos: bool = True, use_rope: str | None = None, num_classes: int = 10):
        super().__init__()
        self.use_pos = use_pos
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)  # initialize cls token
        self.ff = nn.Sequential(nn.Linear(P, dim), nn.GELU())

        # Stack attention blocks
        self.attn_blocks = nn.Sequential(
            *[AttnBlock(dim, n_heads, use_rope=use_rope) for _ in range(k)]
        )

        # Linear layer to project to class logits
        self.ff2 = nn.Sequential(nn.GELU(), nn.Linear(dim, num_classes))

        # Layer for positional encoding. N = grid_size ** 2, the +1 is the cls token
        self.pos = nn.Parameter(torch.randn(1, N + 1, dim) * 0.02)

    def forward(self, seq, out_type: str = "cls", return_features: bool = False):
        """
        Args:
            seq (B, N, P): P is patch_size**2 and N is number of patch rows x columns
            out_type: "cls" | "mean_pool" | "max_pool"

        Returns:
            predicted_labels (B, num_classes), or the pooled features (B, dim)
            when return_features is True
        """
        B, N, P = seq.size()

        # embed sequence
        x = self.ff(seq)

        # prepend cls token to embedded seq
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, dim), expand doesn't copy
        x = torch.cat([cls_token, x], dim=1)          # shape becomes (B, N+1, dim)

        # positional encoding
        if self.use_pos:
            x = x + self.pos

        # Attention blocks
        x = self.attn_blocks(x)

        if out_type == "cls":
            x = x[:, 0]                       # cls token, which theoretically summarizes the rest
        elif out_type == "mean_pool":
            x = x[:, 1:].mean(dim=1)          # mean pool averages all tokens to (B, dim)
        elif out_type == "max_pool":
            x = x[:, 1:].max(dim=1).values    # takes max value (B, dim)

        if return_features:
            return x

        predicted_labels = self.ff2(x)
        return predicted_labels


# --------------------------------------------------------------------------------------
# E5. Linear probe on frozen latents
# --------------------------------------------------------------------------------------

class Classifier(nn.Module):
    """Flattens (B, N, z_dim) latents and classifies them with a small MLP."""

    def __init__(self, z_dim: int, n_tokens: int = 16, hidden: int = 64, num_classes: int = 10):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(z_dim * n_tokens, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        x = einops.rearrange(x, "... n z -> ... (n z)")
        x = self.ff(x)
        return x


# --------------------------------------------------------------------------------------
# 2. Gaussian NLL head - the decoder predicts a mean AND its own uncertainty
# --------------------------------------------------------------------------------------

class ViT_DecoderGaussian(nn.Module):
    def __init__(self, dim: int, n_heads: int, z_dim: int, P: int, N: int, k: int = 2,
                 use_pos: bool = False, use_rope: str | None = "2d"):
        super().__init__()

        # Linear layer to decode from a lower dim to a higher one
        self.ff = nn.Sequential(nn.GELU(), nn.Linear(z_dim, dim))

        # Stack attention blocks
        self.attn_blocks = nn.Sequential(
            *[AttnBlock(dim, n_heads, use_rope=use_rope) for _ in range(k)]
        )

        # *2 for outputting the pixel prediction AND the uncertainty
        self.ff2 = nn.Sequential(nn.GELU(), nn.Linear(dim, P * 2))

        # Layer for positional encoding
        self.pos = nn.Parameter(torch.randn(1, N, dim) * 0.02)  # N = grid_size ** 2
        self.use_pos = use_pos

    def forward(self, seq):
        """
        Args:
            seq (B, N, z_dim): latent vector encoded via ViT_Encoder

        Returns:
            x_mean (B, N, P), x_logvar (B, N, P)
        """
        # embed sequence
        x = self.ff(seq)

        # positional encoding
        if self.use_pos:
            x = x + self.pos

        x = self.attn_blocks(x)

        # project sequence to uncompressed seq
        x = self.ff2(x)

        # Split into mean (average prediction) and logvar (uncertainty in the prediction).
        # We predict log-variance rather than sigma^2 to keep it positive and stable.
        x_mean, x_logvar = x.chunk(2, dim=-1)
        # Clamp for stability: min var exp(-7) ~ 1e-3, max exp(7) ~ 1e3.
        # It did MUCH worse without clamping on FashionMNIST.
        x_logvar = torch.clamp(x_logvar, -7, 7)

        return x_mean, x_logvar


class ViT_AutoEncoderGaussian(nn.Module):
    """
    Train with:
        loss = 0.5 * ((patches - mean).pow(2) * torch.exp(-logvar) + logvar).mean()
    which is equivalent to F.gaussian_nll_loss(mean, patches, torch.exp(logvar)).
    """

    def __init__(self, dim: int, n_heads: int, z_dim: int, patch_size: int, grid_size: int,
                 k: int = 2, use_pos: bool = False, use_rope: str | None = "2d"):
        super().__init__()

        self.patch_size = patch_size
        self.grid_size = grid_size

        self.encoder = ViT_Encoder(dim, n_heads, z_dim, patch_size ** 2, grid_size ** 2,
                                   k=k, use_pos=use_pos, use_rope=use_rope)
        self.decoder = ViT_DecoderGaussian(dim, n_heads, z_dim, patch_size ** 2, grid_size ** 2,
                                           k=k, use_pos=use_pos, use_rope=use_rope)

    def forward(self, images):
        """
        Args:
            images (B, C, H, W)
        """
        image_patches = patchify(images, self.patch_size)
        z = self.encoder(image_patches)
        x_mean, x_logvar = self.decoder(z)
        recon = unpatchify(x_mean, self.patch_size, self.grid_size)

        return recon, x_mean, x_logvar, z


# --------------------------------------------------------------------------------------
# 2.2 Categorical head - bin the pixel range and predict a distribution per pixel
# --------------------------------------------------------------------------------------

class ViT_DecoderCategorical(nn.Module):
    def __init__(self, dim: int, n_heads: int, z_dim: int, P: int, N: int, num_bins: int,
                 k: int = 2, use_pos: bool = False, use_rope: str | None = "2d"):
        super().__init__()

        # Linear layer to decode from a lower dim to a higher one
        self.ff = nn.Sequential(nn.GELU(), nn.Linear(z_dim, dim))

        # Stack attention blocks
        self.attn_blocks = nn.Sequential(
            *[AttnBlock(dim, n_heads, use_rope=use_rope) for _ in range(k)]
        )

        # Unembed to one score per bin per pixel
        self.ff2 = nn.Sequential(nn.GELU(), nn.Linear(dim, P * num_bins))
        self.num_bins = num_bins

        # Layer for positional encoding
        self.pos = nn.Parameter(torch.randn(1, N, dim) * 0.02)  # N = grid_size ** 2
        self.use_pos = use_pos

    def forward(self, seq):
        """
        Args:
            seq (B, N, z_dim): latent vector encoded via ViT_Encoder

        Returns:
            logits (B, N, P, num_bins) for cross entropy, and x_sample (B, N, P) in [0, 1]
        """
        # embed sequence
        x = self.ff(seq)

        # positional encoding
        if self.use_pos:
            x = x + self.pos

        x = self.attn_blocks(x)

        # project sequence to uncompressed seq
        x = self.ff2(x)

        # reshape x to get num of bins
        logits = einops.rearrange(x, "b n (p k) -> b n p k", k=self.num_bins)

        # Greedy alternative: take the bin with the max score
        #   x_sample = logits.argmax(dim=-1).float() / (self.num_bins - 1)
        # Instead we sample proportionally to bin probability, so a lower-scoring bin
        # can still be picked.
        distribution = torch.distributions.Categorical(logits=logits)
        x_sample = distribution.sample().float() / (self.num_bins - 1)

        return logits, x_sample


class ViT_AutoEncoderCategorical(nn.Module):
    """
    Train with:
        targets = (patches * (num_bins - 1)).round().long()
        loss = F.cross_entropy(logits.reshape(-1, num_bins), targets.reshape(-1))
    """

    def __init__(self, dim: int, n_heads: int, z_dim: int, patch_size: int, grid_size: int,
                 num_bins: int, k: int = 2, use_pos: bool = False,
                 use_rope: str | None = "2d"):
        super().__init__()

        self.patch_size = patch_size
        self.grid_size = grid_size
        self.num_bins = num_bins

        self.encoder = ViT_Encoder(dim, n_heads, z_dim, patch_size ** 2, grid_size ** 2,
                                   k=k, use_pos=use_pos, use_rope=use_rope)
        self.decoder = ViT_DecoderCategorical(dim, n_heads, z_dim, patch_size ** 2,
                                              grid_size ** 2, num_bins, k=k,
                                              use_pos=use_pos, use_rope=use_rope)

    def forward(self, images):
        """
        Args:
            images (B, C, H, W)
        """
        image_patches = patchify(images, self.patch_size)
        z = self.encoder(image_patches)
        logits, x_sample = self.decoder(z)  # (B, N, P, num_bins), (B, N, P)
        recon = unpatchify(x_sample, self.patch_size, self.grid_size)

        return recon, x_sample, logits


# --------------------------------------------------------------------------------------
# 3.1 Temporal transformer: causal attention + next-step prediction
# --------------------------------------------------------------------------------------

class AutoRegressivePredictor(nn.Module):
    """Predicts x[t+1] from x[:t] for a scalar time series, using causal blocks + 1d rope."""

    def __init__(self, dim: int = 256, n_head: int = 6, attn_blocks: int = 2,
                 in_dim: int = 1, out_dim: int = 1):
        super().__init__()
        self.encoder = nn.Linear(in_dim, dim)
        self.attn_blocks = nn.Sequential(
            *[AttnBlock(dim, n_head, is_causal=True, use_rope="1d") for _ in range(attn_blocks)]
        )
        self.decoder = nn.Linear(dim, out_dim)

    def forward(self, x):
        """
        Args:
            x (B, T, in_dim)

        Returns:
            (B, T, out_dim), where position t predicts step t+1
        """
        x = self.encoder(x)
        x = self.attn_blocks(x)
        x = self.decoder(x)
        return x
