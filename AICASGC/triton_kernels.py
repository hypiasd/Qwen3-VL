"""
Triton Kernels for VLM Optimization

This module contains the hand-written kernels used by the manual-only profiles.

Stage mapping:
- Vision prefill: `triton_bilinear_pos_embed`, `triton_vision_qkv_rope_transpose`,
  `triton_layernorm`, `triton_gelu_tanh`
- Language prefill/decode: `triton_fused_rmsnorm_rope`, `triton_rmsnorm`
- Candidate-only kernels not wired into the main path yet:
  `fused_layernorm_linear`, `triton_static_cache_update`, `triton_elementwise_mul`

All kernels have PyTorch fallback implementations for environments without Triton.
"""
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("[triton_kernels] Triton not available, using PyTorch fallback")

try:
    from torch.library import triton_op, wrap_triton
    HAS_TRITON_OP = True
except Exception:
    HAS_TRITON_OP = False


if HAS_TRITON:
    @triton.jit
    def bilinear_pos_embed_kernel(
        pos_embed_ptr,
        output_ptr,
        H, W, D,
        H_new, W_new,
        stride_h, stride_w, stride_d,
        stride_h_new, stride_w_new, stride_d_new,
        BLOCK_D: tl.constexpr,
    ):
        """
        Triton kernel for bilinear position embedding interpolation.
        
        Fuses multiple operations:
        - linspace, meshgrid, indexing
        - multiply, sum, reshape, permute
        
        Based on vLLM PR #37948 which achieved 28% encoder speedup.
        """
        pid_h = tl.program_id(0)
        pid_w = tl.program_id(1)
        pid_d = tl.program_id(2)
        
        h_scale = tl.cast(H - 1, tl.float32) / tl.cast(tl.maximum(H_new - 1, 1), tl.float32)
        w_scale = tl.cast(W - 1, tl.float32) / tl.cast(tl.maximum(W_new - 1, 1), tl.float32)
        
        h_in = tl.cast(pid_h, tl.float32) * h_scale
        w_in = tl.cast(pid_w, tl.float32) * w_scale
        
        h0 = tl.floor(h_in).to(tl.int32)
        w0 = tl.floor(w_in).to(tl.int32)
        h1 = tl.minimum(h0 + 1, H - 1)
        w1 = tl.minimum(w0 + 1, W - 1)
        
        h_alpha = h_in - h0
        w_alpha = w_in - w0
        
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        
        p00 = tl.load(pos_embed_ptr + h0 * stride_h + w0 * stride_w + offs_d * stride_d,
                      mask=offs_d < D, other=0.0)
        p01 = tl.load(pos_embed_ptr + h0 * stride_h + w1 * stride_w + offs_d * stride_d,
                      mask=offs_d < D, other=0.0)
        p10 = tl.load(pos_embed_ptr + h1 * stride_h + w0 * stride_w + offs_d * stride_d,
                      mask=offs_d < D, other=0.0)
        p11 = tl.load(pos_embed_ptr + h1 * stride_h + w1 * stride_w + offs_d * stride_d,
                      mask=offs_d < D, other=0.0)
        
        p0 = p00 * (1 - w_alpha) + p01 * w_alpha
        p1 = p10 * (1 - w_alpha) + p11 * w_alpha
        result = p0 * (1 - h_alpha) + p1 * h_alpha
        
        tl.store(output_ptr + pid_h * stride_h_new + pid_w * stride_w_new + offs_d * stride_d_new,
                 result, mask=offs_d < D)

    @triton.jit
    def layernorm_linear_kernel(
        X_ptr, LN_W_ptr, LN_B_ptr, W_ptr, B_ptr, Y_ptr,
        Mean_ptr, Rstd_ptr,
        N, M, K,
        eps,
        stride_x_n, stride_x_m,
        stride_ln_w_m, stride_ln_b_m,
        stride_w_k, stride_w_m,
        stride_y_n, stride_y_k,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Fused LayerNorm + Linear kernel.
        
        Reduces memory bandwidth by fusing:
        1. LayerNorm computation
        2. Linear projection
        
        This eliminates one full memory read/write cycle.
        """
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)

        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        sum_x = tl.zeros([BLOCK_M], dtype=tl.float32)
        sum_sq = tl.zeros([BLOCK_M], dtype=tl.float32)
        block_count = 0

        for m_start in range(0, M, BLOCK_M):
            offs_m = m_start + tl.arange(0, BLOCK_M)
            x_ptrs = X_ptr + pid_n * stride_x_n + offs_m * stride_x_m
            x = tl.load(x_ptrs, mask=offs_m < M, other=0.0).to(tl.float32)
            sum_x += x
            sum_sq += x * x
            block_count += 1

        mean = tl.sum(sum_x, axis=0) / M
        second_moment = tl.sum(sum_sq, axis=0) / M
        var = second_moment - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)

        tl.store(Mean_ptr + pid_n, mean)
        tl.store(Rstd_ptr + pid_n, rstd)

        acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        for m_start in range(0, M, BLOCK_M):
            offs_m = m_start + tl.arange(0, BLOCK_M)
            x_ptrs = X_ptr + pid_n * stride_x_n + offs_m * stride_x_m
            x = tl.load(x_ptrs, mask=offs_m < M, other=0.0).to(tl.float32)
            ln_w = tl.load(LN_W_ptr + offs_m * stride_ln_w_m, mask=offs_m < M, other=1.0).to(tl.float32)
            ln_b = tl.load(LN_B_ptr + offs_m * stride_ln_b_m, mask=offs_m < M, other=0.0).to(tl.float32)
            x_norm = (x - mean) * rstd
            x_norm = x_norm * ln_w + ln_b

            w_ptrs = W_ptr + offs_k[:, None] * stride_w_k + offs_m[None, :] * stride_w_m
            w = tl.load(w_ptrs, mask=(offs_k[:, None] < K) & (offs_m[None, :] < M), other=0.0).to(tl.float32)
            acc += tl.sum(w * x_norm[None, :], axis=1)

        if HAS_BIAS:
            b = tl.load(B_ptr + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
            acc += b

        y_ptrs = Y_ptr + pid_n * stride_y_n + offs_k * stride_y_k
        tl.store(y_ptrs, acc, mask=offs_k < K)

    @triton.jit
    def static_cache_update_kernel(
        cache_ptr, states_ptr,
        B, H, L, D,
        pos,
        stride_cb, stride_ch, stride_cl, stride_cd,
        stride_sb, stride_sh, stride_sl, stride_sd,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_d = tl.program_id(2)

        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

        states_ptrs = (
            states_ptr
            + pid_b * stride_sb
            + pid_h * stride_sh
            + offs_d * stride_sd
        )
        values = tl.load(states_ptrs, mask=(pid_b < B) & (pid_h < H) & (offs_d < D), other=0.0)

        cache_ptrs = (
            cache_ptr
            + pid_b * stride_cb
            + pid_h * stride_ch
            + pos * stride_cl
            + offs_d * stride_cd
        )
        tl.store(cache_ptrs, values, mask=(pid_b < B) & (pid_h < H) & (offs_d < D))

    @triton.jit
    def elementwise_mul_kernel(
        x_ptr, y_ptr, out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, x * y, mask=mask)


def _triton_bilinear_pos_embed_impl(pos_embed: torch.Tensor, H_new: int, W_new: int) -> torch.Tensor:
    """Triton implementation of bilinear position embedding interpolation."""
    H, W, D = pos_embed.shape
    output = torch.empty(H_new, W_new, D, dtype=pos_embed.dtype, device=pos_embed.device)
    
    BLOCK_D = 128
    grid = (H_new, W_new, triton.cdiv(D, BLOCK_D))
    
    bilinear_pos_embed_kernel[grid](
        pos_embed, output,
        H, W, D,
        H_new, W_new,
        pos_embed.stride(0), pos_embed.stride(1), pos_embed.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_D=min(BLOCK_D, D)
    )
    
    return output


def _pytorch_bilinear_pos_embed(pos_embed: torch.Tensor, H_new: int, W_new: int) -> torch.Tensor:
    """PyTorch fallback for bilinear position embedding interpolation."""
    import torch.nn.functional as F
    
    H, W, D = pos_embed.shape
    
    pos_embed_4d = pos_embed.unsqueeze(0).permute(0, 3, 1, 2)
    
    output_4d = F.interpolate(
        pos_embed_4d,
        size=(H_new, W_new),
        mode='bilinear',
        align_corners=False
    )
    
    output = output_4d.squeeze(0).permute(1, 2, 0)
    
    return output


def triton_bilinear_pos_embed(pos_embed: torch.Tensor, H_new: int, W_new: int) -> torch.Tensor:
    # Vision-prefill helper used by the visual position embedding fast path.
    """
    Bilinear position embedding interpolation.
    
    Args:
        pos_embed: [H, W, D] Original position embedding
        H_new, W_new: Target dimensions
    
    Returns:
        [H_new, W_new, D] Interpolated position embedding
    
    Uses Triton kernel if available, otherwise falls back to PyTorch.
    """
    if HAS_TRITON:
        try:
            return _triton_bilinear_pos_embed_impl(pos_embed, H_new, W_new)
        except Exception as e:
            print(f"[triton_kernels] Warning: Triton kernel failed, using PyTorch: {e}")
            return _pytorch_bilinear_pos_embed(pos_embed, H_new, W_new)
    else:
        return _pytorch_bilinear_pos_embed(pos_embed, H_new, W_new)


def _triton_layernorm_linear_impl(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """Triton implementation of fused LayerNorm + Linear."""
    N, M = x.shape
    K = weight.shape[0]
    
    output = torch.empty(N, K, dtype=x.dtype, device=x.device)
    mean = torch.empty(N, dtype=x.dtype, device=x.device)
    rstd = torch.empty(N, dtype=x.dtype, device=x.device)
    
    BLOCK_M = 128
    BLOCK_K = 128
    has_bias = bias is not None
    bias_tensor = bias if has_bias else weight
    grid = (N, triton.cdiv(K, BLOCK_K))
    
    layernorm_linear_kernel[grid](
        x, norm_weight, norm_bias, weight, bias_tensor, output,
        mean, rstd,
        N, M, K, eps,
        x.stride(0), x.stride(1),
        norm_weight.stride(0), norm_bias.stride(0),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        HAS_BIAS=has_bias,
        BLOCK_M=min(BLOCK_M, M),
        BLOCK_K=min(BLOCK_K, K)
    )
    
    return output


def _pytorch_layernorm_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """PyTorch fallback for fused LayerNorm + Linear."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    x_norm = x_norm * norm_weight + norm_bias
    
    output = torch.nn.functional.linear(x_norm, weight, bias)
    
    return output


def fused_layernorm_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    # Candidate kernel for language-side hot paths. Not yet wired into evaluation_wrapper.py.
    """
    Fused LayerNorm + Linear operation.
    
    Args:
        x: [N, M] Input tensor
        weight: [K, M] Linear layer weight
        bias: [K] Linear layer bias
        eps: LayerNorm epsilon
    
    Returns:
        [N, K] Output tensor
    
    Uses Triton kernel if available, otherwise falls back to PyTorch.
    """
    if HAS_TRITON:
        try:
            return _triton_layernorm_linear_impl(x, norm_weight, norm_bias, weight, bias, eps)
        except Exception as e:
            print(f"[triton_kernels] Warning: Triton kernel failed, using PyTorch: {e}")
            return _pytorch_layernorm_linear(x, norm_weight, norm_bias, weight, bias, eps)
    else:
        return _pytorch_layernorm_linear(x, norm_weight, norm_bias, weight, bias, eps)


def _triton_static_cache_update_impl(
    cache: torch.Tensor,
    states: torch.Tensor,
    position: int
) -> torch.Tensor:
    B, H, L, D = cache.shape
    BLOCK_D = 128
    grid = (B, H, triton.cdiv(D, BLOCK_D))
    static_cache_update_kernel[grid](
        cache, states,
        B, H, L, D,
        position,
        cache.stride(0), cache.stride(1), cache.stride(2), cache.stride(3),
        states.stride(0), states.stride(1), states.stride(2), states.stride(3),
        BLOCK_D=min(BLOCK_D, D),
    )
    return cache


def triton_static_cache_update(
    cache: torch.Tensor,
    states: torch.Tensor,
    position: int
) -> torch.Tensor:
    # Candidate handwritten KV update path. Not used by the current manual profiles.
    if HAS_TRITON_OP and hasattr(torch.ops, "aicas_ops") and hasattr(torch.ops.aicas_ops, "static_cache_update"):
        torch.ops.aicas_ops.static_cache_update(cache, states, position)
        return cache
    if HAS_TRITON:
        try:
            return _triton_static_cache_update_impl(cache, states, position)
        except Exception as e:
            print(f"[triton_kernels] Warning: Triton cache update failed, using PyTorch: {e}")
            cache[:, :, position:position + 1, :].copy_(states)
            return cache
    cache[:, :, position:position + 1, :].copy_(states)
    return cache


def _triton_elementwise_mul_impl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    elementwise_mul_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_elementwise_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Candidate elementwise helper. Keep out of the main path until profiling shows a need.
    if HAS_TRITON_OP and hasattr(torch.ops, "aicas_ops") and hasattr(torch.ops.aicas_ops, "elementwise_mul"):
        return torch.ops.aicas_ops.elementwise_mul(x, y)
    if HAS_TRITON:
        try:
            return _triton_elementwise_mul_impl(x, y)
        except Exception as e:
            print(f"[triton_kernels] Warning: Triton mul failed, using PyTorch: {e}")
            return x * y
    return x * y


if HAS_TRITON and HAS_TRITON_OP:
    @triton_op("aicas_ops::static_cache_update", mutates_args={"cache"})
    def _triton_static_cache_update_op(cache: torch.Tensor, states: torch.Tensor, position: int) -> None:
        B, H, L, D = cache.shape
        BLOCK_D = 128
        grid = (B, H, triton.cdiv(D, BLOCK_D))
        wrap_triton(static_cache_update_kernel)[grid](
            cache, states,
            B, H, L, D,
            position,
            cache.stride(0), cache.stride(1), cache.stride(2), cache.stride(3),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3),
            BLOCK_D=min(BLOCK_D, D),
        )

    @triton_op("aicas_ops::elementwise_mul", mutates_args={})
    def _triton_elementwise_mul_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n_elements = out.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        wrap_triton(elementwise_mul_kernel)[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out



__all__ = [
    'triton_bilinear_pos_embed',
    'fused_layernorm_linear',
    'triton_static_cache_update',
    'triton_elementwise_mul',
    'HAS_TRITON',
    'triton_silu_mul',
]

if HAS_TRITON:
    @triton.jit
    def fused_rmsnorm_rope_kernel(
        x_ptr, weight_ptr,
        cos_ptr, sin_ptr,
        out_ptr,
        B, L, H, D,
        eps,
        stride_x_b, stride_x_l, stride_x_h, stride_x_d,
        stride_out_b, stride_out_h, stride_out_l, stride_out_d,
        stride_cos_b, stride_cos_l, stride_cos_d,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        pid_h = tl.program_id(2)
        
        offs_d = tl.arange(0, BLOCK_D)
        mask = offs_d < D
        
        x_ptrs = x_ptr + pid_b * stride_x_b + pid_l * stride_x_l + pid_h * stride_x_h + offs_d * stride_x_d
        x = tl.load(x_ptrs, mask=mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
        
        var = tl.sum(x * x, axis=0) / D
        rsqrt = tl.math.rsqrt(var + eps)
        
        w = tl.load(weight_ptr + offs_d, mask=mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        x_norm = x * rsqrt * w
        
        cos_ptrs = cos_ptr + pid_b * stride_cos_b + pid_l * stride_cos_l + offs_d * stride_cos_d
        sin_ptrs = sin_ptr + pid_b * stride_cos_b + pid_l * stride_cos_l + offs_d * stride_cos_d
        
        cos = tl.load(cos_ptrs, mask=mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        sin = tl.load(sin_ptrs, mask=mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        
        half_D = D // 2
        rot_offs_d = (offs_d + half_D) % D
        x_rot_ptrs = x_ptr + pid_b * stride_x_b + pid_l * stride_x_l + pid_h * stride_x_h + rot_offs_d * stride_x_d
        x_rot = tl.load(x_rot_ptrs, mask=mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
        w_rot = tl.load(weight_ptr + rot_offs_d, mask=mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        x_norm_rot = x_rot * rsqrt * w_rot
        
        sign = tl.where(offs_d < half_D, -1.0, 1.0)
        x_norm_rot = x_norm_rot * sign
        
        out = x_norm * cos + x_norm_rot * sin
        
        out_ptrs = out_ptr + pid_b * stride_out_b + pid_h * stride_out_h + pid_l * stride_out_l + offs_d * stride_out_d
        tl.store(out_ptrs, out.to(x_ptr.dtype.element_ty), mask=mask)

if HAS_TRITON and HAS_TRITON_OP:
    @triton_op("aicas_ops::fused_rmsnorm_rope", mutates_args={})
    def triton_fused_rmsnorm_rope(
        x: torch.Tensor, 
        weight: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor, 
        eps: float
    ) -> torch.Tensor:
        # Main language-model kernel for the decode-heavy path.
        B, L, H, D = x.shape
        out = torch.empty((B, H, L, D), dtype=x.dtype, device=x.device)
        BLOCK_D = triton.next_power_of_2(D)
        grid = (B, L, H)
        wrap_triton(fused_rmsnorm_rope_kernel)[grid](
            x, weight, cos, sin, out,
            B, L, H, D, eps,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            cos.stride(0), cos.stride(1), cos.stride(2),
            BLOCK_D=BLOCK_D,
            num_warps=4,
        )
        return out

if HAS_TRITON:
    @triton.jit
    def rmsnorm_kernel(
        x_ptr, weight_ptr, out_ptr,
        stride_x_row, stride_out_row,
        N, eps,
        BLOCK_N: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        x_ptrs = x_ptr + row_idx * stride_x_row + tl.arange(0, BLOCK_N)
        mask = tl.arange(0, BLOCK_N) < N
        
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        rsqrt = tl.math.rsqrt(var + eps)
        
        w = tl.load(weight_ptr + tl.arange(0, BLOCK_N), mask=mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        out = x * rsqrt * w
        
        out_ptrs = out_ptr + row_idx * stride_out_row + tl.arange(0, BLOCK_N)
        tl.store(out_ptrs, out.to(x_ptr.dtype.element_ty), mask=mask)

if HAS_TRITON and HAS_TRITON_OP:
    @triton_op("aicas_ops::rmsnorm", mutates_args={})
    def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        # Main language-model norm replacement used in both prefill and decode.
        x_2d = x.view(-1, x.shape[-1])
        out_2d = torch.empty_like(x_2d)
        N = x.shape[-1]
        M = x_2d.shape[0]
        
        BLOCK_N = triton.next_power_of_2(N)
        grid = (M,)
        wrap_triton(rmsnorm_kernel)[grid](
            x_2d, weight, out_2d,
            x_2d.stride(0), out_2d.stride(0),
            N, eps,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out_2d.view_as(x)


if HAS_TRITON:
    @triton.jit
    def fused_vision_qkv_rope_transpose_kernel(
        qkv_ptr,
        cos_ptr, sin_ptr,
        q_out_ptr, k_out_ptr, v_out_ptr,
        S, H, D,
        stride_qkv_s, stride_qkv_d,
        stride_out_h, stride_out_s, stride_out_d,
        stride_cos_s, stride_cos_d,
        BLOCK_D: tl.constexpr
    ):
        pid_s = tl.program_id(0)
        pid_h = tl.program_id(1)

        offs_d = tl.arange(0, BLOCK_D)
        mask = offs_d < D

        q_offset = pid_s * stride_qkv_s + 0 * (H * D) + pid_h * D + offs_d
        k_offset = pid_s * stride_qkv_s + 1 * (H * D) + pid_h * D + offs_d
        v_offset = pid_s * stride_qkv_s + 2 * (H * D) + pid_h * D + offs_d

        q = tl.load(qkv_ptr + q_offset, mask=mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
        k = tl.load(qkv_ptr + k_offset, mask=mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
        v = tl.load(qkv_ptr + v_offset, mask=mask, other=0.0, eviction_policy='evict_first').to(tl.float32)

        cos = tl.load(cos_ptr + pid_s * stride_cos_s + offs_d * stride_cos_d, mask=mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        sin = tl.load(sin_ptr + pid_s * stride_cos_s + offs_d * stride_cos_d, mask=mask, other=0.0, eviction_policy='evict_last').to(tl.float32)

        half_D = D // 2
        rot_offs_d = (offs_d + half_D) % D
        
        q_rot_offset = pid_s * stride_qkv_s + 0 * (H * D) + pid_h * D + rot_offs_d
        k_rot_offset = pid_s * stride_qkv_s + 1 * (H * D) + pid_h * D + rot_offs_d

        q_rot = tl.load(qkv_ptr + q_rot_offset, mask=mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
        k_rot = tl.load(qkv_ptr + k_rot_offset, mask=mask, other=0.0, eviction_policy='evict_first').to(tl.float32)

        sign = tl.where(offs_d < half_D, -1.0, 1.0)
        q_rot = q_rot * sign
        k_rot = k_rot * sign

        q_rope = q * cos + q_rot * sin
        k_rope = k * cos + k_rot * sin

        out_offset = pid_h * stride_out_h + pid_s * stride_out_s + offs_d * stride_out_d
        tl.store(q_out_ptr + out_offset, q_rope.to(qkv_ptr.dtype.element_ty), mask=mask)
        tl.store(k_out_ptr + out_offset, k_rope.to(qkv_ptr.dtype.element_ty), mask=mask)
        tl.store(v_out_ptr + out_offset, v.to(qkv_ptr.dtype.element_ty), mask=mask)

if HAS_TRITON and HAS_TRITON_OP:
    @triton_op("aicas_ops::fused_vision_qkv_rope_transpose", mutates_args={})
    def triton_vision_qkv_rope_transpose(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, H: int, D: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Main vision-prefill fused kernel, not part of the iterative decode loop.
        S = qkv.shape[0]
        q_out = torch.empty((1, H, S, D), dtype=qkv.dtype, device=qkv.device)
        k_out = torch.empty((1, H, S, D), dtype=qkv.dtype, device=qkv.device)
        v_out = torch.empty((1, H, S, D), dtype=qkv.dtype, device=qkv.device)

        BLOCK_D = triton.next_power_of_2(D)
        grid = (S, H)
        wrap_triton(fused_vision_qkv_rope_transpose_kernel)[grid](
            qkv, cos, sin,
            q_out, k_out, v_out,
            S, H, D,
            qkv.stride(0), qkv.stride(1),
            q_out.stride(1), q_out.stride(2), q_out.stride(3),
            cos.stride(0), cos.stride(1),
            BLOCK_D=BLOCK_D,
            num_warps=4,
        )
        return q_out, k_out, v_out


if HAS_TRITON:
    @triton.jit
    def fused_layernorm_kernel(
        x_ptr, weight_ptr, bias_ptr, out_ptr,
        stride_x_row, stride_out_row,
        N, eps,
        BLOCK_N: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        x_ptrs = x_ptr + row_idx * stride_x_row + tl.arange(0, BLOCK_N)
        mask = tl.arange(0, BLOCK_N) < N
        
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        
        mean = tl.sum(x, axis=0) / N
        x_centered = x - mean
        var = tl.sum(x_centered * x_centered, axis=0) / N
        rsqrt = tl.math.rsqrt(var + eps)
        
        w = tl.load(weight_ptr + tl.arange(0, BLOCK_N), mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + tl.arange(0, BLOCK_N), mask=mask, other=0.0).to(tl.float32)
        
        out = x_centered * rsqrt * w + b
        
        out_ptrs = out_ptr + row_idx * stride_out_row + tl.arange(0, BLOCK_N)
        tl.store(out_ptrs, out.to(x_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def fused_gelu_tanh_kernel(
        x_ptr, out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        x = tl.load(x_ptr + offs, mask=mask, other=0.0, eviction_policy='evict_first').to(tl.float32)

        c1 = 0.7978845608 # sqrt(2/pi)
        c2 = 0.044715
        inner = c1 * (x + c2 * x * x * x)
        
        exp2x = tl.exp(2.0 * inner)
        tanh_inner = (exp2x - 1.0) / (exp2x + 1.0)
        tanh_inner = tl.where(inner > 10.0, 1.0, tanh_inner)
        tanh_inner = tl.where(inner < -10.0, -1.0, tanh_inner)
        
        out = x * 0.5 * (1.0 + tanh_inner)

        tl.store(out_ptr + offs, out.to(x_ptr.dtype.element_ty), mask=mask)

if HAS_TRITON and HAS_TRITON_OP:
    @triton_op("aicas_ops::layernorm", mutates_args={})
    def triton_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
        # Vision-prefill LayerNorm replacement.
        x_2d = x.view(-1, x.shape[-1])
        out_2d = torch.empty_like(x_2d)
        N = x.shape[-1]
        M = x_2d.shape[0]
        
        BLOCK_N = triton.next_power_of_2(N)
        grid = (M,)
        wrap_triton(fused_layernorm_kernel)[grid](
            x_2d, weight, bias, out_2d,
            x_2d.stride(0), out_2d.stride(0),
            N, eps,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out_2d.view_as(x)

    @triton_op("aicas_ops::gelu_tanh", mutates_args={})
    def triton_gelu_tanh(x: torch.Tensor) -> torch.Tensor:
        # Vision-prefill activation replacement.
        out = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        wrap_triton(fused_gelu_tanh_kernel)[grid](
            x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=4
        )
        return out

if HAS_TRITON:
    @triton.jit
    def fused_residual_rmsnorm_kernel(
        X_ptr,          # [M, N] The current hidden states
        Residual_ptr,   # [M, N] The residual connection
        NormW_ptr,      # [N] The RMSNorm weights
        Out_ptr,        # [M, N] The output after RMSNorm
        ResOut_ptr,     # [M, N] The updated residual (X + Residual)
        M, N,
        stride_xm, stride_xn,
        stride_resm, stride_resn,
        stride_outm, stride_outn,
        stride_resoutm, stride_resoutn,
        EPS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_idx = pid
        if row_idx >= M:
            return
            
        X_row_ptr = X_ptr + row_idx * stride_xm
        Res_row_ptr = Residual_ptr + row_idx * stride_resm
        Out_row_ptr = Out_ptr + row_idx * stride_outm
        ResOut_row_ptr = ResOut_ptr + row_idx * stride_resoutm
        
        offsets = tl.arange(0, BLOCK_N)
        mask = offsets < N
        
        # evict_first for x since it's only read once and never reused
        x = tl.load(X_row_ptr + offsets * stride_xn, mask=mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
        # res is read and written, let's keep it in L2
        res = tl.load(Res_row_ptr + offsets * stride_resn, mask=mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        
        x_res = x + res
        
        variance = tl.sum(x_res * x_res, axis=0) / N
        rsqrt = tl.math.rsqrt(variance + EPS)
        
        # norm_w is read multiple times across rows, keep it in L2
        norm_w = tl.load(NormW_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        out = x_res * rsqrt * norm_w
        
        tl.store(ResOut_row_ptr + offsets * stride_resoutn, x_res.to(X_ptr.dtype.element_ty), mask=mask)
        tl.store(Out_row_ptr + offsets * stride_outn, out.to(X_ptr.dtype.element_ty), mask=mask)

    def triton_fused_residual_rmsnorm(x, residual, norm_weight, eps=1e-6):
        orig_shape = x.shape
        x_2d = x.view(-1, orig_shape[-1])
        res_2d = residual.view(-1, orig_shape[-1])
        
        M, N = x_2d.shape
        out = torch.empty_like(x_2d)
        BLOCK_N = triton.next_power_of_2(N)
        
        grid = (M,)
        fused_residual_rmsnorm_kernel[grid](
            x_2d, res_2d, norm_weight, out, res_2d,
            M, N,
            x_2d.stride(0), x_2d.stride(1),
            res_2d.stride(0), res_2d.stride(1),
            out.stride(0), out.stride(1),
            res_2d.stride(0), res_2d.stride(1),
            EPS=eps,
            BLOCK_N=BLOCK_N,
            num_warps=4 if BLOCK_N <= 2048 else 8
        )
        
        return out.view(orig_shape)

if HAS_TRITON:
    @triton.jit
    def _silu_mul_kernel(
        gate_up_ptr,
        out_ptr,
        n_elements,
        intermediate_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        row = offsets // intermediate_size
        col = offsets % intermediate_size
        
        gate_idx = row * (2 * intermediate_size) + col
        up_idx = gate_idx + intermediate_size
        
        gate = tl.load(gate_up_ptr + gate_idx, mask=mask, eviction_policy='evict_first')
        up = tl.load(gate_up_ptr + up_idx, mask=mask, eviction_policy='evict_first')
        
        gate_f32 = gate.to(tl.float32)
        silu_gate = gate_f32 * tl.sigmoid(gate_f32)
        res = (silu_gate.to(gate.dtype)) * up
        
        tl.store(out_ptr + offsets, res, mask=mask)

    def triton_silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
        shape = gate_up.shape
        intermediate_size = shape[-1] // 2
        out_shape = list(shape)
        out_shape[-1] = intermediate_size
        out = torch.empty(out_shape, device=gate_up.device, dtype=gate_up.dtype)
        n_elements = out.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _silu_mul_kernel[grid](gate_up, out, n_elements, intermediate_size, BLOCK_SIZE=1024)
        return out
else:
    def triton_silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
        shape = gate_up.shape
        intermediate_size = shape[-1] // 2
        gate, up = torch.split(gate_up, intermediate_size, dim=-1)
        return torch.nn.functional.silu(gate) * up
