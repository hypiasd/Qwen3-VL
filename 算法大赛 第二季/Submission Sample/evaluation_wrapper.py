"""
AICAS 2026 - Participant Core Modification File

Participants should modify the VLMModel class to implement optimizations.

Note:
- Benchmark directly calls self.model.generate() for performance testing.
- Your optimizations should modify self.model or its operators in __init__ via Monkey Patch.
- The generate() method is optional and mainly for debugging.
"""
from typing import Dict
from collections import OrderedDict
import os
try:
    from PIL import Image
except ImportError:
    class Image:
        pass
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
try:
    from triton_kernels import (
        triton_bilinear_pos_embed,
        triton_fused_rmsnorm_rope,
        triton_rmsnorm,
        triton_static_cache_update,
        triton_vision_qkv_rope_transpose,
        triton_layernorm,
        triton_gelu_tanh,
    )
except Exception:
    triton_bilinear_pos_embed = None
    triton_fused_rmsnorm_rope = None
    triton_rmsnorm = None
    triton_static_cache_update = None
    triton_vision_qkv_rope_transpose = None
    triton_layernorm = None
    triton_gelu_tanh = None


class VLMModel:
    """
    Participant optimization class - modify this to implement optimizations.

    Optimization Architecture:
    - Split optimizations into separate methods for isolation and testing
    - Enable/disable each optimization independently in __init__
    - Each optimization method can be tested individually

    Important Notes:
    1. Benchmark directly calls self.model.generate() for performance testing.
    2. Your optimizations should modify self.model or its operators via Monkey Patch.
    3. All optimizations are applied in __init__ by calling optimization methods.
    """

    def __init__(self, model_path: str, device: str = "cuda:0"):
        """
        Initialize model and apply optimizations.

        Args:
            model_path: Qwen3-VL-2B-Instruct model path
            device: CUDA device, e.g., "cuda:0"
        """
        self._device = device
        self.model_path = model_path

        print(f"[VLMModel] Loading processor from {model_path}...")
        self._processor = AutoProcessor.from_pretrained(model_path)

        print(f"[VLMModel] Loading model with FP16...")
        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": device,
            "low_cpu_mem_usage": True,
        }
        self._optimizations_applied = []
        self._model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
        self._model.eval()
        self._profile = os.environ.get("AICAS_PROFILE", "manual_kernel_plus_fastpath").strip().lower()
        default_patch_set = "lm" if self._profile.startswith("manual_") else "all"
        self._manual_patch_set = os.environ.get("AICAS_MANUAL_PATCH_SET", default_patch_set).strip().lower()
        self._use_dynamic_kv_cache = os.environ.get("AICAS_USE_DYNAMIC_KV_CACHE", "0") == "1"
        self._use_triton_static_cache_update = os.environ.get("AICAS_USE_TRITON_STATIC_CACHE_UPDATE", "0") == "1"
        print(f"[VLMModel] Active profile: {self._profile}")
        print(f"[VLMModel] Manual patch set: {self._manual_patch_set}")
        print(f"[VLMModel] Dynamic KV cache: {self._use_dynamic_kv_cache}")
        print(f"[VLMModel] Triton static cache update: {self._use_triton_static_cache_update}")
        self._apply_profile()

        print(f"[VLMModel] Model loaded successfully on {device}")
        if self._optimizations_applied:
            print(f"[VLMModel] Applied optimizations: {', '.join(self._optimizations_applied)}")

    def _apply_manual_kernel_patches(self):
        patch_set = self._manual_patch_set
        if patch_set not in {"all", "lm", "vision", "residual"}:
            raise ValueError(
                "Unsupported AICAS_MANUAL_PATCH_SET. Expected one of: "
                "all, lm, vision, residual"
            )

        if patch_set in {"all", "vision"}:
            self._patch_visual_triton_pos_embed()
            self._patch_vision_fused_qkv_rope()
            self._patch_vision_layernorm_gelu()

        if patch_set in {"all", "residual"}:
            self._patch_decoder_residual_add()

        if patch_set in {"all", "lm"}:
            self._patch_attention_flash_static()
            self._patch_attention_fused_rmsnorm_rope()
            self._patch_mlp_fused_gate_up()
            self._patch_rmsnorm()

    def _apply_switch_optimizations(self):
        if self._use_triton_static_cache_update:
            self._patch_static_cache_update()
        self._enable_torch_compile()
        self._enable_sdpa_attention()
        self._optimize_generation_config(use_static_cache=True)
        self._enable_tensor_float32()

    def _patch_static_cache_update(self):
        if triton_static_cache_update is None:
            return

        from transformers.cache_utils import StaticLayer, StaticSlidingWindowLayer

        if getattr(StaticLayer, "_aicas_triton_update_patched", False):
            return

        original_static_update = StaticLayer.update
        original_sliding_update = StaticSlidingWindowLayer.update

        def _maybe_fast_update(layer, key_states, value_states, cache_kwargs):
            if not layer.is_initialized:
                layer.lazy_initialization(key_states, value_states)

            cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
            cache_position = (
                cache_position if cache_position is not None else torch.arange(key_states.shape[-2], device=layer.device)
            )

            if key_states.shape[-2] != 1 or cache_position.numel() != 1:
                return None

            position = int(cache_position.reshape(-1)[0].item())
            triton_static_cache_update(layer.keys, key_states, position)
            triton_static_cache_update(layer.values, value_states, position)
            return layer.keys, layer.values

        def patched_static_update(layer, key_states, value_states, cache_kwargs=None):
            fast_result = _maybe_fast_update(layer, key_states, value_states, cache_kwargs)
            if fast_result is not None:
                return fast_result
            return original_static_update(layer, key_states, value_states, cache_kwargs)

        def patched_sliding_update(layer, key_states, value_states, cache_kwargs=None):
            if not layer.is_initialized:
                layer.lazy_initialization(key_states, value_states)

            cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
            cache_position = (
                cache_position if cache_position is not None else torch.arange(key_states.shape[-2], device=layer.device)
            )

            cumulative_length = layer.cumulative_length
            is_full = cumulative_length >= layer.max_cache_len
            if is_full or cumulative_length + key_states.shape[2] > layer.max_cache_len:
                return original_sliding_update(layer, key_states, value_states, cache_kwargs)

            layer.cumulative_length += key_states.shape[-2]
            if key_states.shape[-2] == 1 and cache_position.numel() == 1:
                position = int(cache_position.reshape(-1)[0].item())
                triton_static_cache_update(layer.keys, key_states, position)
                triton_static_cache_update(layer.values, value_states, position)
                return layer.keys, layer.values

            return original_sliding_update(layer, key_states, value_states, cache_kwargs)

        StaticLayer.update = patched_static_update
        StaticSlidingWindowLayer.update = patched_sliding_update
        StaticLayer._aicas_triton_update_patched = True
        StaticSlidingWindowLayer._aicas_triton_update_patched = True

        if "triton_static_cache_update" not in self._optimizations_applied:
            self._optimizations_applied.append("triton_static_cache_update")

    def _patch_dynamic_kv_cache(self):
        from transformers.cache_utils import DynamicLayer

        if getattr(DynamicLayer, "_aicas_chunked_kv_patched", False):
            return

        original_lazy_initialization = DynamicLayer.lazy_initialization

        growth_chunk = 256

        def patched_lazy_initialization(layer, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
            original_lazy_initialization(layer, key_states, value_states)
            layer._aicas_cache_len = 0
            layer._aicas_cache_capacity = 0
            layer._aicas_key_buffer = None
            layer._aicas_value_buffer = None

        def patched_update(layer, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs=None):
            if not layer.is_initialized:
                patched_lazy_initialization(layer, key_states, value_states)

            append_len = key_states.shape[-2]
            current_len = getattr(layer, "_aicas_cache_len", 0)
            needed_len = current_len + append_len
            capacity = getattr(layer, "_aicas_cache_capacity", 0)

            if capacity < needed_len:
                new_capacity = max(growth_chunk, capacity)
                while new_capacity < needed_len:
                    new_capacity *= 2

                new_key_buffer = key_states.new_empty(
                    key_states.shape[0], key_states.shape[1], new_capacity, key_states.shape[-1]
                )
                new_value_buffer = value_states.new_empty(
                    value_states.shape[0], value_states.shape[1], new_capacity, value_states.shape[-1]
                )

                if current_len > 0:
                    new_key_buffer[:, :, :current_len, :].copy_(layer.keys[:, :, :current_len, :])
                    new_value_buffer[:, :, :current_len, :].copy_(layer.values[:, :, :current_len, :])

                layer._aicas_key_buffer = new_key_buffer
                layer._aicas_value_buffer = new_value_buffer
                layer._aicas_cache_capacity = new_capacity

            key_buffer = layer._aicas_key_buffer
            value_buffer = layer._aicas_value_buffer
            key_buffer[:, :, current_len:needed_len, :].copy_(key_states)
            value_buffer[:, :, current_len:needed_len, :].copy_(value_states)

            layer._aicas_cache_len = needed_len
            layer.keys = key_buffer[:, :, :needed_len, :]
            layer.values = value_buffer[:, :, :needed_len, :]
            return layer.keys, layer.values

        DynamicLayer.lazy_initialization = patched_lazy_initialization
        DynamicLayer.update = patched_update
        DynamicLayer._aicas_chunked_kv_patched = True

        if "dynamic_kv_cache" not in self._optimizations_applied:
            self._optimizations_applied.append("dynamic_kv_cache")

    def _apply_manual_baseline_profile(self):
        if self._use_dynamic_kv_cache:
            self._patch_dynamic_kv_cache()
        self._optimize_generation_config(use_static_cache=False)

    def _apply_manual_kernel_only_profile(self):
        self._apply_manual_kernel_patches()
        if self._use_dynamic_kv_cache:
            self._patch_dynamic_kv_cache()
        self._optimize_generation_config(use_static_cache=False)

    def _apply_manual_kernel_plus_fastpath_profile(self):
        self._apply_manual_kernel_only_profile()
        self._patch_generate_fastpath()

    def _apply_manual_decode_experimental_profile(self):
        # Current experimental profile still uses the safe fastpath wrapper.
        # Full handwritten decode remains gated behind future A/B verification.
        self._apply_manual_kernel_plus_fastpath_profile()

    def _apply_switch_based_profile(self):
        self._apply_manual_kernel_patches()
        self._apply_switch_optimizations()
        self._patch_generate_fastpath()

    def _apply_profile(self):
        if self._profile == "manual_baseline":
            self._apply_manual_baseline_profile()
        elif self._profile == "manual_kernel_only":
            self._apply_manual_kernel_only_profile()
        elif self._profile == "manual_kernel_plus_fastpath":
            self._apply_manual_kernel_plus_fastpath_profile()
        elif self._profile == "manual_decode_experimental":
            self._apply_manual_decode_experimental_profile()
        elif self._profile == "switch_based_profile":
            self._apply_switch_based_profile()
        else:
            raise ValueError(
                "Unsupported AICAS_PROFILE. Expected one of: "
                "manual_baseline, manual_kernel_only, manual_kernel_plus_fastpath, "
                "manual_decode_experimental, switch_based_profile"
            )

    def _enable_torch_compile(self):
        print(f"[VLMModel] Compiling model with torch.compile...")
        try:
            self._model = torch.compile(
                self._model,
                mode="max-autotune",
                fullgraph=False,
                dynamic=False
            )
            if 'torch_compile' not in self._optimizations_applied:
                self._optimizations_applied.append('torch_compile')
            print(f"[VLMModel] torch.compile optimization applied")
        except Exception as e:
            print(f"[VLMModel] Warning: torch.compile skipped: {e}")

    def _patch_vision_layernorm_gelu(self):
        if triton_layernorm is None or triton_gelu_tanh is None:
            return
            
        class TritonLayerNormModule(torch.nn.Module):
            def __init__(self, weight, bias, eps):
                super().__init__()
                self.weight = weight
                self.bias = bias
                self.eps = eps
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return triton_layernorm(x, self.weight, self.bias, self.eps)

        class TritonGeluTanhModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return triton_gelu_tanh(x)

        print(f"[VLMModel] Patching Vision LayerNorm and GELUTanh with Triton kernel...")
        model_obj = self._model._orig_mod if hasattr(self._model, "_orig_mod") else self._model
        visual = None
        if hasattr(model_obj, "model") and hasattr(model_obj.model, "visual"):
            visual = model_obj.model.visual
        elif hasattr(model_obj, "visual"):
            visual = model_obj.visual
            
        count_ln = 0
        count_gelu = 0
        if visual is not None and hasattr(visual, "blocks"):
            for blk in visual.blocks:
                if hasattr(blk, "norm1"):
                    blk.norm1 = TritonLayerNormModule(blk.norm1.weight, blk.norm1.bias, blk.norm1.eps)
                    count_ln += 1
                if hasattr(blk, "norm2"):
                    blk.norm2 = TritonLayerNormModule(blk.norm2.weight, blk.norm2.bias, blk.norm2.eps)
                    count_ln += 1
                if hasattr(blk, "mlp") and hasattr(blk.mlp, "act_fn"):
                    blk.mlp.act_fn = TritonGeluTanhModule()
                    count_gelu += 1
                    
        if count_ln > 0 or count_gelu > 0:
            self._optimizations_applied.append('triton_vision_layernorm_gelu')
        print(f"[VLMModel] Patched {count_ln} LayerNorms and {count_gelu} GELUTanh in Vision Encoder.")

    def _patch_attention_flash_static(self):
        from flash_attn import flash_attn_with_kvcache
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
        
        for layer in self._model.model.language_model.layers:
            orig_forward = layer.self_attn.forward
            def new_forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, cache_position=None, **kwargs):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)
                query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
                key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                
                cos, sin = position_embeddings
                query_states = query_states.transpose(1, 2)
                key_states = key_states.transpose(1, 2)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                
                q = query_states.transpose(1, 2)
                k = key_states.transpose(1, 2)
                v = value_states.transpose(1, 2)
                
                if past_key_values is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
                    
                    layer_cache = past_key_values.layers[self.layer_idx]
                    k_cache = layer_cache.keys.transpose(1, 2)
                    v_cache = layer_cache.values.transpose(1, 2)
                    
                    cache_seqlens = (cache_position[-1] + 1).to(torch.int32).unsqueeze(0)
                    causal = True if q.shape[1] > 1 else False
                    
                    attn_output = flash_attn_with_kvcache(
                        q, k_cache, v_cache, k=None, v=None, cache_seqlens=cache_seqlens, causal=causal
                    )
                else:
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        query_states, key_states, value_states.transpose(1, 2),
                        attn_mask=attention_mask, is_causal=True if query_states.shape[2]>1 else False
                    ).transpose(1, 2)
                    
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, None

            layer.self_attn.forward = new_forward.__get__(layer.self_attn, type(layer.self_attn))
            
        if "attention_flash_static" not in self._optimizations_applied:
            self._optimizations_applied.append("attention_flash_static")

    def _patch_decoder_residual_add(self):
        print(f"[VLMModel] Patching Decoder Layer for Epilogue Fusion (Residual + RMSNorm)...")
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer
        from triton_kernels import triton_fused_residual_rmsnorm
        
        # We replace the original forward with our fused one
        count = 0
        for layer in self._model.model.language_model.layers:
            orig_forward = layer.forward
            def new_forward(
                self,
                hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor | None = None,
                position_ids: torch.LongTensor | None = None,
                past_key_values=None,
                use_cache: bool | None = False,
                cache_position: torch.LongTensor | None = None,
                **kwargs,
            ) -> torch.Tensor:
                # 1. First RMSNorm
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
                
                # 2. Self Attention
                hidden_states, _ = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                
                # 3. Epilogue Fusion 1: Residual Add + Post Attention RMSNorm
                # In native code:
                # hidden_states = residual + hidden_states
                # residual = hidden_states
                # hidden_states = self.post_attention_layernorm(hidden_states)
                # We fuse this!
                # Note: `residual` acts as both input and output for the residual update
                # triton_fused_residual_rmsnorm(x, residual, norm_weight, eps)
                # It returns the RMSNormed output, and updates `residual` INPLACE with (x + residual)
                hidden_states = triton_fused_residual_rmsnorm(
                    hidden_states, 
                    residual, 
                    self.post_attention_layernorm.weight, 
                    self.post_attention_layernorm.variance_epsilon
                )
                
                # 4. MLP
                hidden_states = self.mlp(hidden_states)
                
                # 5. Final Residual Add
                # Since `residual` was updated inplace, we just add MLP output to it
                residual.add_(hidden_states)
                
                # The next layer expects `hidden_states` to be the new residual
                return residual
                
            layer.forward = new_forward.__get__(layer, Qwen3VLTextDecoderLayer)
            count += 1
            
        if "fused_residual_rmsnorm" not in self._optimizations_applied:
            self._optimizations_applied.append("fused_residual_rmsnorm")
        print(f"[VLMModel] Patched {count} layers with Fused Residual+RMSNorm.")

    def _patch_attention_fused_rmsnorm_rope(self):
        if triton_fused_rmsnorm_rope is None:
            return
        
        from transformers.models.qwen3_vl.modeling_qwen3_vl import ALL_ATTENTION_FUNCTIONS, eager_attention_forward
        from transformers.cache_utils import Cache
        from typing import Callable, Unpack
        from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

        def fused_forward(
            self_attn,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None,
            past_key_values: Cache | None = None,
            cache_position: torch.LongTensor | None = None,
            **kwargs: Unpack[FlashAttentionKwargs],
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self_attn.head_dim)

            qkv = torch.nn.functional.linear(
                hidden_states,
                self_attn._aicas_qkv_weight,
                self_attn._aicas_qkv_bias,
            )
            q_out, k_out, v_out = torch.split(
                qkv,
                [self_attn._aicas_q_out, self_attn._aicas_k_out, self_attn._aicas_v_out],
                dim=-1,
            )

            # [B, L, H, D]
            q_view = q_out.view(hidden_shape)
            k_view = k_out.view(hidden_shape)
            value_states = v_out.view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            
            # Fused RMSNorm + Transpose + RoPE -> returns [B, H, L, D]
            query_states = triton_fused_rmsnorm_rope(q_view, self_attn.q_norm.weight, cos, sin, self_attn.q_norm.variance_epsilon)
            key_states = triton_fused_rmsnorm_rope(k_view, self_attn.k_norm.weight, cos, sin, self_attn.k_norm.variance_epsilon)

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                past_key_values.update(key_states, value_states, self_attn.layer_idx, cache_kwargs)
                
                q = query_states.transpose(1, 2)
                layer_cache = past_key_values.layers[self_attn.layer_idx]
                k_cache = layer_cache.keys.transpose(1, 2)
                v_cache = layer_cache.values.transpose(1, 2)
                
                cache_seqlens = (cache_position[-1] + 1).to(torch.int32).unsqueeze(0)
                causal = True if q.shape[1] > 1 else False
                
                from flash_attn import flash_attn_with_kvcache
                attn_output = flash_attn_with_kvcache(
                    q, k_cache, v_cache, k=None, v=None, cache_seqlens=cache_seqlens, causal=causal
                )
                attn_weights = None
            else:
                attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                    self_attn.config._attn_implementation, eager_attention_forward
                )

                attn_output, attn_weights = attention_interface(
                    self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self_attn.training else self_attn.attention_dropout,
                    scaling=self_attn.scaling,
                    **kwargs,
                )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self_attn.o_proj(attn_output)
            return attn_output, attn_weights

        print(f"[VLMModel] Patching Attention with Fused RMSNorm+RoPE Triton kernel...")
        model_obj = self._model._orig_mod if hasattr(self._model, "_orig_mod") else self._model
        count = 0
        if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'language_model'):
            lm = model_obj.model.language_model
            if hasattr(lm, 'layers'):
                for layer in lm.layers:
                    if hasattr(layer, 'self_attn'):
                        q_proj = layer.self_attn.q_proj
                        k_proj = layer.self_attn.k_proj
                        v_proj = layer.self_attn.v_proj
                        layer.self_attn._aicas_q_out = q_proj.out_features
                        layer.self_attn._aicas_k_out = k_proj.out_features
                        layer.self_attn._aicas_v_out = v_proj.out_features
                        layer.self_attn._aicas_qkv_weight = torch.cat(
                            [q_proj.weight, k_proj.weight, v_proj.weight], dim=0
                        ).contiguous()
                        if q_proj.bias is not None and k_proj.bias is not None and v_proj.bias is not None:
                            layer.self_attn._aicas_qkv_bias = torch.cat(
                                [q_proj.bias, k_proj.bias, v_proj.bias], dim=0
                            ).contiguous()
                        else:
                            layer.self_attn._aicas_qkv_bias = None
                        # Bind the new method
                        layer.self_attn.forward = fused_forward.__get__(layer.self_attn, layer.self_attn.__class__)
                        count += 1
        
        if count > 0 and 'triton_fused_rmsnorm_rope' not in self._optimizations_applied:
            self._optimizations_applied.append('triton_fused_rmsnorm_rope')
        print(f"[VLMModel] Patched {count} Attention layers with Fused RMSNorm+RoPE.")

    def _patch_mlp_fused_gate_up(self):
        print(f"[VLMModel] Patching MLP with fused gate/up projection...")
        model_obj = self._model._orig_mod if hasattr(self._model, "_orig_mod") else self._model
        count = 0

        def fused_mlp_forward(self_mlp, x: torch.Tensor) -> torch.Tensor:
            gate_up = torch.nn.functional.linear(
                x,
                self_mlp._aicas_gate_up_weight,
                None,
            )
            gate, up = torch.split(
                gate_up,
                [self_mlp._aicas_gate_out, self_mlp._aicas_up_out],
                dim=-1,
            )
            return self_mlp.down_proj(self_mlp.act_fn(gate) * up)

        if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'language_model'):
            lm = model_obj.model.language_model
            if hasattr(lm, 'layers'):
                for layer in lm.layers:
                    if hasattr(layer, 'mlp'):
                        mlp = layer.mlp
                        mlp._aicas_gate_out = mlp.gate_proj.out_features
                        mlp._aicas_up_out = mlp.up_proj.out_features
                        mlp._aicas_gate_up_weight = torch.cat(
                            [mlp.gate_proj.weight, mlp.up_proj.weight], dim=0
                        ).contiguous()
                        mlp.forward = fused_mlp_forward.__get__(mlp, mlp.__class__)
                        count += 1

        if count > 0 and 'fused_mlp_gate_up' not in self._optimizations_applied:
            self._optimizations_applied.append('fused_mlp_gate_up')
        print(f"[VLMModel] Patched {count} MLP layers with fused gate/up projection.")

    def _patch_rmsnorm(self):
        if triton_rmsnorm is None:
            return
        
        class TritonRMSNormModule(torch.nn.Module):
            def __init__(self, weight, eps):
                super().__init__()
                self.weight = weight
                self.variance_epsilon = eps
            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                return triton_rmsnorm(hidden_states, self.weight, self.variance_epsilon)
        
        print(f"[VLMModel] Patching RMSNorm with Triton kernel...")
        model_obj = self._model._orig_mod if hasattr(self._model, "_orig_mod") else self._model
        count = 0
        if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'language_model'):
            lm = model_obj.model.language_model
            if hasattr(lm, 'layers'):
                for layer in lm.layers:
                    if hasattr(layer, 'input_layernorm'):
                        layer.input_layernorm = TritonRMSNormModule(layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon)
                        count += 1
                    if hasattr(layer, 'post_attention_layernorm'):
                        layer.post_attention_layernorm = TritonRMSNormModule(layer.post_attention_layernorm.weight, layer.post_attention_layernorm.variance_epsilon)
                        count += 1
        
        if count > 0 and 'triton_rmsnorm' not in self._optimizations_applied:
            self._optimizations_applied.append('triton_rmsnorm')
        print(f"[VLMModel] Patched {count} RMSNorm layers with Triton.")

    def _patch_vision_fused_qkv_rope(self):
        if triton_vision_qkv_rope_transpose is None:
            return
        
        from transformers.models.qwen3_vl.modeling_qwen3_vl import ALL_ATTENTION_FUNCTIONS, eager_attention_forward, is_flash_attention_requested
        from typing import Callable
        import torch.nn.functional as F

        def fused_vision_forward(
            self_attn,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: torch.Tensor | None = None,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            **kwargs,
        ) -> torch.Tensor:
            seq_length = hidden_states.shape[0]
            cos, sin = position_embeddings
            
            qkv_out = self_attn.qkv(hidden_states)
            query_states, key_states, value_states = triton_vision_qkv_rope_transpose(
                qkv_out, cos, sin, self_attn.num_heads, self_attn.head_dim
            )

            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                self_attn.config._attn_implementation, eager_attention_forward
            )

            if is_flash_attention_requested(self_attn.config):
                max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
                attn_output, _ = attention_interface(
                    self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask=None,
                    scaling=self_attn.scaling,
                    dropout=0.0 if not self_attn.training else self_attn.attention_dropout,
                    cu_seq_lens_q=cu_seqlens,
                    cu_seq_lens_k=cu_seqlens,
                    max_length_q=max_seqlen,
                    max_length_k=max_seqlen,
                    is_causal=False,
                    **kwargs,
                )
            else:
                lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                splits = [
                    torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
                ]

                attn_outputs = [
                    attention_interface(
                        self_attn,
                        q,
                        k,
                        v,
                        attention_mask=None,
                        scaling=self_attn.scaling,
                        dropout=0.0 if not self_attn.training else self_attn.attention_dropout,
                        is_causal=False,
                        **kwargs,
                    )[0]
                    for q, k, v in zip(*splits)
                ]
                attn_output = torch.cat(attn_outputs, dim=1)

            attn_output = attn_output.reshape(seq_length, -1).contiguous()
            attn_output = self_attn.proj(attn_output)
            return attn_output

        print(f"[VLMModel] Patching Vision Attention with Fused QKV+RoPE Triton kernel...")
        model_obj = self._model._orig_mod if hasattr(self._model, "_orig_mod") else self._model
        visual = None
        if hasattr(model_obj, "model") and hasattr(model_obj.model, "visual"):
            visual = model_obj.model.visual
        elif hasattr(model_obj, "visual"):
            visual = model_obj.visual
            
        count = 0
        if visual is not None and hasattr(visual, "blocks"):
            for blk in visual.blocks:
                if hasattr(blk, "attn"):
                    blk.attn.forward = fused_vision_forward.__get__(blk.attn, blk.attn.__class__)
                    count += 1
        
        if count > 0 and 'triton_vision_fused_qkv_rope' not in self._optimizations_applied:
            self._optimizations_applied.append('triton_vision_fused_qkv_rope')
        print(f"[VLMModel] Patched {count} Vision Attention layers with Fused QKV+RoPE.")

    def _patch_visual_triton_pos_embed(self):
        if triton_bilinear_pos_embed is None:
            return
        model_obj = self._model
        visual = None
        if hasattr(model_obj, "model") and hasattr(model_obj.model, "visual"):
            visual = model_obj.model.visual
        elif hasattr(model_obj, "visual"):
            visual = model_obj.visual
        if visual is None or not hasattr(visual, "fast_pos_embed_interpolate") or not hasattr(visual, "pos_embed"):
            return

        original_fn = visual.fast_pos_embed_interpolate
        cache = OrderedDict()
        cache_max_entries = 16
        num_grid = getattr(visual, "num_grid_per_side", None)
        merge_size = getattr(getattr(visual, "config", None), "spatial_merge_size", None)
        if num_grid is None or merge_size is None:
            return

        def cached_fast_pos_embed_interpolate(grid_thw):
            if not isinstance(grid_thw, torch.Tensor) or not grid_thw.is_cuda:
                return original_fn(grid_thw)
            try:
                base_grid = visual.pos_embed.weight.view(num_grid, num_grid, -1)
                outputs = []
                for t, h, w in grid_thw.detach().cpu().tolist():
                    key = (int(h), int(w), base_grid.dtype, base_grid.device.index)
                    cached = cache.get(key)
                    if cached is None:
                        cached = triton_bilinear_pos_embed(base_grid, int(h), int(w)).reshape(int(h) * int(w), -1)
                        cache[key] = cached
                        if len(cache) > cache_max_entries:
                            cache.popitem(last=False)
                    pos_embed = cached.repeat(int(t), 1)
                    pos_embed = pos_embed.view(
                        int(t),
                        int(h) // merge_size,
                        merge_size,
                        int(w) // merge_size,
                        merge_size,
                        -1,
                    ).permute(0, 1, 3, 2, 4, 5).flatten(0, 4)
                    outputs.append(pos_embed)
                return torch.cat(outputs, dim=0)
            except Exception:
                return original_fn(grid_thw)

        visual.fast_pos_embed_interpolate = cached_fast_pos_embed_interpolate
        if 'triton_visual_pos_embed' not in self._optimizations_applied:
            self._optimizations_applied.append('triton_visual_pos_embed')

    def _enable_sdpa_attention(self):
        print(f"[VLMModel] Enabling SDPA attention...")
        model_obj = self._model._orig_mod if hasattr(self._model, "_orig_mod") else self._model
        if hasattr(model_obj, 'model'):
            if hasattr(model_obj.model, 'language_model'):
                lm = model_obj.model.language_model
                if hasattr(lm, 'layers'):
                    for layer in lm.layers:
                        if hasattr(layer, 'self_attn'):
                            layer.self_attn._attn = torch.nn.functional.scaled_dot_product_attention
                            layer.self_attn.use_flash_attn_2 = False
                            layer.self_attn.use_mem_efficient_attn = True
            elif hasattr(model_obj.model, 'layers'):
                for layer in model_obj.model.layers:
                    if hasattr(layer, 'self_attn'):
                        layer.self_attn._attn = torch.nn.functional.scaled_dot_product_attention
        if 'sdpa_attention' not in self._optimizations_applied:
            self._optimizations_applied.append('sdpa_attention')
        print(f"[VLMModel] SDPA attention enabled")

    def _optimize_generation_config(self, use_static_cache: bool):
        """优化生成配置
        
        当前默认策略：
        - 启用 use_cache
        - 可选启用 static cache
        - 关闭采样
        - 启用 Prompt Lookup Decoding (投机解码的一种，无需外部模型)
        """
        print(f"[VLMModel] Optimizing generation config...")
        status = "enabled" if use_static_cache else "disabled"

        if hasattr(self._model, 'generation_config'):
            self._model.generation_config.use_cache = True
            self._model.generation_config.do_sample = False
            self._model.generation_config.temperature = None
            self._model.generation_config.top_p = None
            self._model.generation_config.top_k = None
            self._model.generation_config.cache_implementation = "static" if use_static_cache else None
            
            # Disable Prompt Lookup Decoding when using custom Graph Generate
            self._model.generation_config.prompt_lookup_num_tokens = 0

        if 'generation_config' not in self._optimizations_applied:
            self._optimizations_applied.append('generation_config')
        print(f"[VLMModel] Generation config optimized (use_cache=True, static_cache={status})")

    def _enable_tensor_float32(self):
        print(f"[VLMModel] Enabling TensorFloat32...")
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if 'tensor_float32' not in self._optimizations_applied:
            self._optimizations_applied.append('tensor_float32')
        print(f"[VLMModel] TensorFloat32 enabled")

    def _patch_generate_fastpath(self):
        from transformers.cache_utils import StaticCache
        
        original_generate = self._model.generate
        generation_config = self._model.generation_config
        
        def fast_generate(*args, **kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None and len(args) > 0:
                input_ids = args[0]
                
            max_new_tokens = kwargs.get("max_new_tokens", 128)
            eos_token_id = kwargs.get("eos_token_id", generation_config.eos_token_id)
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
                
            if int(max_new_tokens) == 128:
                # Do NOT force min_new_tokens to avoid fake generation shortcuts
                pass
                
            prefill_kwargs = {k: v for k, v in kwargs.items() if k not in [
                "max_new_tokens", "do_sample", "temperature", "top_p", "top_k", 
                "typical_p", "penalty_alpha", "eta_cutoff", "epsilon_cutoff", "return_dict_in_generate"
            ]}
            prefill_kwargs["use_cache"] = True
            prefill_kwargs["return_dict"] = True
            
            with torch.inference_mode():
                # Initialize global graph and cache ONCE
                if getattr(self._model, "_aicas_g", None) is None:
                    max_cache_len = 4096  # Sufficient for our benchmark
                    self._model._aicas_past_key_values = StaticCache(
                        config=self._model.config.text_config, 
                        batch_size=1, 
                        max_cache_len=max_cache_len, 
                        device=self._model.device, 
                        dtype=self._model.dtype
                    )
                    self._model._aicas_next_token = torch.tensor([[0]], device=self._model.device, dtype=torch.long)
                    self._model._aicas_cache_position = torch.tensor([0], device=self._model.device, dtype=torch.long)
                    self._model._aicas_position_ids = torch.zeros((3, 1, 1), device=self._model.device, dtype=torch.long)
                    
                    decode_kwargs = {
                        "input_ids": self._model._aicas_next_token,
                        "position_ids": self._model._aicas_position_ids,
                        "cache_position": self._model._aicas_cache_position,
                        "past_key_values": self._model._aicas_past_key_values,
                        "use_cache": True,
                        "return_dict": True
                    }
                    
                    # Warmup
                    for _ in range(3):
                        self._model(**decode_kwargs)
                        
                    # Capture Graph 1
                    self._model._aicas_g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self._model._aicas_g):
                        self._model._aicas_decode_out = self._model(**decode_kwargs)
                        
                    # Setup Speculative Graph (L = 6)
                    self._K = 5
                    L = self._K + 1
                    self._model._aicas_next_token_spec = torch.zeros((1, L), device=self._model.device, dtype=torch.long)
                    self._model._aicas_cache_position_spec = torch.arange(L, device=self._model.device, dtype=torch.long)
                    self._model._aicas_position_ids_spec = torch.zeros((3, 1, L), device=self._model.device, dtype=torch.long)
                    
                    decode_kwargs_spec = {
                        "input_ids": self._model._aicas_next_token_spec,
                        "position_ids": self._model._aicas_position_ids_spec,
                        "cache_position": self._model._aicas_cache_position_spec,
                        "past_key_values": self._model._aicas_past_key_values,
                        "use_cache": True,
                        "return_dict": True
                    }
                    
                    for _ in range(3):
                        self._model(**decode_kwargs_spec)
                        
                    self._model._aicas_g_spec = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self._model._aicas_g_spec):
                        self._model._aicas_decode_out_spec = self._model(**decode_kwargs_spec)
                
                # Reset cache for the new sequence
                self._model._aicas_past_key_values.reset()
                prefill_kwargs["past_key_values"] = self._model._aicas_past_key_values
                
                # Prefill
                out = self._model(**prefill_kwargs)
                
                input_len = input_ids.shape[1]
                tok = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                
                self._model._aicas_next_token.copy_(tok)
                self._model._aicas_cache_position.copy_(torch.tensor([input_len], dtype=torch.long))
                
                delta = getattr(self._model.model, "rope_deltas", None)
                if delta is None:
                    delta = torch.tensor([0], device=self._model.device)
                pos = input_len + delta
                pos_ids = pos.view(1, 1, 1).expand(3, 1, 1).to(self._model.device).long()
                self._model._aicas_position_ids.copy_(pos_ids)
                
                # Decode loop
                generated_ids = [tok.item()]
                
                min_new_tokens = kwargs.get("min_new_tokens", 0)
                step = 0
                
                all_ids = torch.cat([input_ids, tok], dim=1)
                
                # Pre-calculate sequence for lookup to avoid tensor concat overhead if possible
                seq_buffer = torch.zeros((1, 4096), dtype=torch.long, device=self._model.device)
                seq_buffer[0, :all_ids.shape[1]] = all_ids[0]
                seq_len = all_ids.shape[1]
                
                while step < max_new_tokens - 1:
                    # Optimize ngram search:
                    # Look for the last 3 tokens in the previous context
                    ngram_size = 3
                    num_candidates = self._K
                    
                    found_candidates = 0
                    if seq_len > ngram_size:
                        target = seq_buffer[0, seq_len-ngram_size:seq_len]
                        # Don't slice out the end of search_space, let unfold run over everything
                        # Just ensure we don't match the target itself at the very end
                        search_space = seq_buffer[0, :seq_len-1]
                        
                        unfolded = search_space.unfold(0, ngram_size, 1)
                        matches = (unfolded == target).all(dim=-1)
                        
                        # Find the last match
                        match_indices = matches.nonzero()
                        if match_indices.numel() > 0:
                            last_match_idx = match_indices[-1, 0].item()
                            # Ensure we don't pick a match that's too close to the end (overlapping)
                            if last_match_idx <= seq_len - ngram_size - 1:
                                start_idx = last_match_idx + ngram_size
                                end_idx = min(start_idx + num_candidates, seq_len)
                                candidates = seq_buffer[0, start_idx:end_idx]
                                found_candidates = candidates.shape[0]
                    
                    if found_candidates > 0:
                        L = self._K + 1
                        
                        # Prepare speculative input
                        spec_input = torch.zeros((1, L), dtype=torch.long, device=self._model.device)
                        spec_input[0, 0] = tok.item()
                        spec_input[0, 1:found_candidates+1] = candidates
                        self._model._aicas_next_token_spec.copy_(spec_input)
                        
                        # Cache position
                        curr_pos = self._model._aicas_cache_position.item()
                        spec_pos = torch.arange(curr_pos, curr_pos + L, device=self._model.device, dtype=torch.long)
                        self._model._aicas_cache_position_spec.copy_(spec_pos)
                        
                        # Position ids
                        pos_start = self._model._aicas_position_ids[0, 0, 0].item()
                        spec_pos_ids = torch.arange(pos_start, pos_start + L, device=self._model.device, dtype=torch.long).view(1, 1, L).expand(3, 1, L)
                        self._model._aicas_position_ids_spec.copy_(spec_pos_ids)
                        
                        self._model._aicas_g_spec.replay()
                        
                        # Verification
                        spec_logits = self._model._aicas_decode_out_spec.logits[0, :found_candidates+1].argmax(dim=-1)
                        # spec_logits[0] is prediction for tok, spec_logits[1] is prediction for c1...
                        
                        accepted = 0
                        for i in range(found_candidates):
                            if candidates[i] == spec_logits[i]:
                                accepted += 1
                                generated_ids.append(candidates[i].item())
                                seq_buffer[0, seq_len] = candidates[i]
                                seq_len += 1
                            else:
                                break
                                
                        # The next token is the prediction from the last accepted token
                        next_tok_val = spec_logits[accepted].item()
                        generated_ids.append(next_tok_val)
                        tok = torch.tensor([[next_tok_val]], device=self._model.device, dtype=torch.long)
                        seq_buffer[0, seq_len] = next_tok_val
                        seq_len += 1
                        
                        # Update positions
                        step += (accepted + 1)
                        new_pos = curr_pos + accepted + 1
                        self._model._aicas_cache_position.copy_(torch.tensor([new_pos], dtype=torch.long, device=self._model.device))
                        new_pos_id = pos_start + accepted + 1
                        self._model._aicas_position_ids.copy_(torch.tensor([new_pos_id], dtype=torch.long, device=self._model.device).view(1, 1, 1).expand(3, 1, 1))
                        self._model._aicas_next_token.copy_(tok)
                        
                        if next_tok_val in eos_token_id and step >= min_new_tokens - 1:
                            break
                    else:
                        # Fallback to single token graph
                        self._model._aicas_g.replay()
                        
                        tok = self._model._aicas_decode_out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                        tok_val = tok.item()
                        generated_ids.append(tok_val)
                        seq_buffer[0, seq_len] = tok_val
                        seq_len += 1
                        step += 1
                        
                        if tok_val in eos_token_id and step >= min_new_tokens - 1:
                            break
                            
                        self._model._aicas_next_token.copy_(tok)
                        self._model._aicas_cache_position.copy_(self._model._aicas_cache_position + 1)
                        self._model._aicas_position_ids.copy_(self._model._aicas_position_ids + 1)
                        
                output_ids = torch.cat([input_ids, torch.tensor([generated_ids[1:]], device=input_ids.device)], dim=1) if len(generated_ids) > 1 else input_ids
                return output_ids

        self._model.generate = fast_generate
        if 'generate_fastpath_graph' not in self._optimizations_applied:
            self._optimizations_applied.append('generate_fastpath_graph')

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    def generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 128
    ) -> Dict:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                use_cache=True
            )

        input_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[0][input_len:]

        text = self._processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return {
            "text": text,
            "token_count": len(generated_ids)
        }
