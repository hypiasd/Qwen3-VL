"""
Triton Kernels for VLM Optimization

This module provides optimized Triton kernels for vision encoder operations.
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
        
        h_ratio = H_new / H
        w_ratio = W_new / W
        
        h_in = pid_h / h_ratio
        w_in = pid_w / w_ratio
        
        h0 = int(h_in)
        w0 = int(w_in)
        h1 = min(h0 + 1, H - 1)
        w1 = min(w0 + 1, W - 1)
        
        h_alpha = h_in - h0
        w_alpha = w_in - w0
        
        offs_d = tl.arange(0, BLOCK_D)
        
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
        X_ptr, W_ptr, B_ptr, Y_ptr,
        Mean_ptr, Rstd_ptr,
        N, M, K,
        eps,
        stride_x_n, stride_x_m,
        stride_w_k, stride_w_m,
        stride_y_n, stride_y_k,
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
        
        offs_m = tl.arange(0, BLOCK_M)
        x_ptrs = X_ptr + pid_n * stride_x_n + offs_m * stride_x_m
        
        x = tl.load(x_ptrs, mask=offs_m < M, other=0.0)
        mean = tl.sum(x, axis=0) / M
        
        x_centered = x - mean
        var = tl.sum(x_centered * x_centered, axis=0) / M
        rstd = 1.0 / tl.sqrt(var + eps)
        
        x_norm = x_centered * rstd
        
        tl.store(Mean_ptr + pid_n, mean)
        tl.store(Rstd_ptr + pid_n, rstd)
        
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            
            w_ptrs = W_ptr + offs_k[:, None] * stride_w_k + offs_m[None, :] * stride_w_m
            w = tl.load(w_ptrs, mask=(offs_k[:, None] < K) & (offs_m[None, :] < M), other=0.0)
            
            y = tl.sum(w * x_norm[None, :], axis=1)
            
            if B_ptr is not None:
                b = tl.load(B_ptr + offs_k, mask=offs_k < K, other=0.0)
                y = y + b
            
            y_ptrs = Y_ptr + pid_n * stride_y_n + offs_k * stride_y_k
            tl.store(y_ptrs, y, mask=offs_k < K)


def _triton_bilinear_pos_embed_impl(pos_embed: torch.Tensor, H_new: int, W_new: int) -> torch.Tensor:
    """Triton implementation of bilinear position embedding interpolation."""
    H, W, D = pos_embed.shape
    output = torch.empty(H_new, W_new, D, dtype=pos_embed.dtype, device=pos_embed.device)
    
    BLOCK_D = 128
    grid = (H_new, W_new)
    
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
    grid = (N,)
    
    layernorm_linear_kernel[grid](
        x, weight, bias, output,
        mean, rstd,
        N, M, K, eps,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=min(BLOCK_M, M),
        BLOCK_K=min(BLOCK_K, K)
    )
    
    return output


def _pytorch_layernorm_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """PyTorch fallback for fused LayerNorm + Linear."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    
    output = torch.nn.functional.linear(x_norm, weight, bias)
    
    return output


def fused_layernorm_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
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
            return _triton_layernorm_linear_impl(x, weight, bias, eps)
        except Exception as e:
            print(f"[triton_kernels] Warning: Triton kernel failed, using PyTorch: {e}")
            return _pytorch_layernorm_linear(x, weight, bias, eps)
    else:
        return _pytorch_layernorm_linear(x, weight, bias, eps)


__all__ = [
    'triton_bilinear_pos_embed',
    'fused_layernorm_linear',
    'HAS_TRITON'
]
