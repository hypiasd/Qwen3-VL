"""
Fused Kernels for VLM Optimization

This module provides fused operator implementations for common layer combinations.
All kernels have PyTorch fallback implementations for environments without Triton.
"""
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("[fused_kernels] Triton not available, using PyTorch fallback")


if HAS_TRITON:
    @triton.jit
    def linear_gelu_kernel(
        X_ptr, W_ptr, B_ptr, Y_ptr,
        N, M, K,
        stride_x_n, stride_x_m,
        stride_w_k, stride_w_m,
        stride_y_n, stride_y_k,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Fused Linear + GELU kernel.
        
        Combines:
        1. Linear projection: y = x @ W^T + b
        2. GELU activation: gelu(y)
        
        This reduces kernel launch overhead and memory bandwidth.
        """
        pid_n = tl.program_id(0)
        
        offs_m = tl.arange(0, BLOCK_M)
        x_ptrs = X_ptr + pid_n * stride_x_n + offs_m * stride_x_m
        x = tl.load(x_ptrs, mask=offs_m < M, other=0.0)
        
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            
            w_ptrs = W_ptr + offs_k[:, None] * stride_w_k + offs_m[None, :] * stride_w_m
            w = tl.load(w_ptrs, mask=(offs_k[:, None] < K) & (offs_m[None, :] < M), other=0.0)
            
            y = tl.sum(w * x[None, :], axis=1)
            
            if B_ptr is not None:
                b = tl.load(B_ptr + offs_k, mask=offs_k < K, other=0.0)
                y = y + b
            
            # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            # Using fast approximation
            gelu = 0.5 * y * (1.0 + tl.libdevice.tanh(0.7978845608 * (y + 0.044715 * y * y * y)))
            
            y_ptrs = Y_ptr + pid_n * stride_y_n + offs_k * stride_y_k
            tl.store(y_ptrs, gelu, mask=offs_k < K)

    @triton.jit
    def linear_silu_kernel(
        X_ptr, W_ptr, B_ptr, Y_ptr,
        N, M, K,
        stride_x_n, stride_x_m,
        stride_w_k, stride_w_m,
        stride_y_n, stride_y_k,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Fused Linear + SiLU (Swish) kernel.
        
        Combines:
        1. Linear projection: y = x @ W^T + b
        2. SiLU activation: silu(y) = y * sigmoid(y)
        
        Used in SwiGLU architecture.
        """
        pid_n = tl.program_id(0)
        
        offs_m = tl.arange(0, BLOCK_M)
        x_ptrs = X_ptr + pid_n * stride_x_n + offs_m * stride_x_m
        x = tl.load(x_ptrs, mask=offs_m < M, other=0.0)
        
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            
            w_ptrs = W_ptr + offs_k[:, None] * stride_w_k + offs_m[None, :] * stride_w_m
            w = tl.load(w_ptrs, mask=(offs_k[:, None] < K) & (offs_m[None, :] < M), other=0.0)
            
            y = tl.sum(w * x[None, :], axis=1)
            
            if B_ptr is not None:
                b = tl.load(B_ptr + offs_k, mask=offs_k < K, other=0.0)
                y = y + b
            
            # SiLU(x) = x * sigmoid(x)
            silu = y * tl.sigmoid(y)
            
            y_ptrs = Y_ptr + pid_n * stride_y_n + offs_k * stride_y_k
            tl.store(y_ptrs, silu, mask=offs_k < K)


def _triton_linear_gelu_impl(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Triton implementation of fused Linear + GELU."""
    N, M = x.shape
    K = weight.shape[0]
    
    output = torch.empty(N, K, dtype=x.dtype, device=x.device)
    
    BLOCK_M = 128
    BLOCK_K = 128
    grid = (N,)
    
    linear_gelu_kernel[grid](
        x, weight, bias, output,
        N, M, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=min(BLOCK_M, M),
        BLOCK_K=min(BLOCK_K, K)
    )
    
    return output


def _triton_linear_silu_impl(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Triton implementation of fused Linear + SiLU."""
    N, M = x.shape
    K = weight.shape[0]
    
    output = torch.empty(N, K, dtype=x.dtype, device=x.device)
    
    BLOCK_M = 128
    BLOCK_K = 128
    grid = (N,)
    
    linear_silu_kernel[grid](
        x, weight, bias, output,
        N, M, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=min(BLOCK_M, M),
        BLOCK_K=min(BLOCK_K, K)
    )
    
    return output


def fused_linear_gelu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused Linear + GELU operation.
    
    Args:
        x: [N, M] Input tensor
        weight: [K, M] Linear layer weight
        bias: [K] Linear layer bias (can be None)
    
    Returns:
        [N, K] Output tensor after linear + GELU
    
    Uses Triton kernel if available, otherwise falls back to PyTorch.
    """
    if HAS_TRITON:
        try:
            return _triton_linear_gelu_impl(x, weight, bias)
        except Exception as e:
            print(f"[fused_kernels] Warning: Triton kernel failed, using PyTorch: {e}")
            return torch.nn.functional.gelu(
                torch.nn.functional.linear(x, weight, bias)
            )
    else:
        return torch.nn.functional.gelu(
            torch.nn.functional.linear(x, weight, bias)
        )


def fused_linear_silu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused Linear + SiLU (Swish) operation.
    
    Args:
        x: [N, M] Input tensor
        weight: [K, M] Linear layer weight
        bias: [K] Linear layer bias (can be None)
    
    Returns:
        [N, K] Output tensor after linear + SiLU
    
    Uses Triton kernel if available, otherwise falls back to PyTorch.
    """
    if HAS_TRITON:
        try:
            return _triton_linear_silu_impl(x, weight, bias)
        except Exception as e:
            print(f"[fused_kernels] Warning: Triton kernel failed, using PyTorch: {e}")
            return torch.nn.functional.silu(
                torch.nn.functional.linear(x, weight, bias)
            )
    else:
        return torch.nn.functional.silu(
            torch.nn.functional.linear(x, weight, bias)
        )


__all__ = [
    'fused_linear_gelu',
    'fused_linear_silu',
    'HAS_TRITON'
]
