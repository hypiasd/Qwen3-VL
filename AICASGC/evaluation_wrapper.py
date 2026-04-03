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
try:
    from PIL import Image
except ImportError:
    class Image:
        pass
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
try:
    from triton_kernels import triton_bilinear_pos_embed, triton_fused_rmsnorm_rope, triton_rmsnorm, triton_vision_qkv_rope_transpose, triton_layernorm, triton_gelu_tanh
except Exception:
    triton_bilinear_pos_embed = None
    triton_fused_rmsnorm_rope = None
    triton_rmsnorm = None
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

        self._patch_visual_triton_pos_embed()
        self._patch_vision_fused_qkv_rope()
        self._patch_vision_layernorm_gelu()
        self._patch_decoder_residual_add()
        self._patch_attention_fused_rmsnorm_rope()
        self._patch_rmsnorm()
        self._enable_torch_compile()
        self._enable_sdpa_attention()
        self._optimize_generation_config()
        self._enable_tensor_float32()
        self._patch_generate_fastpath()

        print(f"[VLMModel] Model loaded successfully on {device}")
        if self._optimizations_applied:
            print(f"[VLMModel] Applied optimizations: {', '.join(self._optimizations_applied)}")

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

    def _patch_decoder_residual_add(self):
        print(f"[VLMModel] Patching Decoder Layer for inplace residual add...")
        model_obj = self._model._orig_mod if hasattr(self._model, "_orig_mod") else self._model
        
        count = 0
        if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'language_model'):
            lm = model_obj.model.language_model
            if hasattr(lm, 'layers'):
                for layer in lm.layers:
                    original_forward = layer.forward
                    
                    def new_forward(
                        self_layer,
                        hidden_states: torch.Tensor,
                        position_embeddings: tuple[torch.Tensor, torch.Tensor],
                        attention_mask: torch.Tensor | None = None,
                        position_ids: torch.LongTensor | None = None,
                        past_key_values: tuple[torch.Tensor] | None = None,
                        use_cache: bool | None = False,
                        cache_position: torch.LongTensor | None = None,
                        **kwargs,
                    ):
                        residual = hidden_states
                        hidden_states = self_layer.input_layernorm(hidden_states)
                        hidden_states, _ = self_layer.self_attn(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_values=past_key_values,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            **kwargs,
                        )
                        hidden_states = residual.add_(hidden_states)
                        
                        residual = hidden_states
                        hidden_states = self_layer.post_attention_layernorm(hidden_states)
                        hidden_states = self_layer.mlp(hidden_states)
                        hidden_states = residual.add_(hidden_states)
                        
                        return hidden_states
                        
                    layer.forward = new_forward.__get__(layer, layer.__class__)
                    count += 1
                    
        if count > 0 and 'decoder_residual_add' not in self._optimizations_applied:
            self._optimizations_applied.append('decoder_residual_add')
        print(f"[VLMModel] Patched {count} Decoder layers with inplace residual add.")

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

            # [B, L, H, D]
            q_view = self_attn.q_proj(hidden_states).view(hidden_shape)
            k_view = self_attn.k_proj(hidden_states).view(hidden_shape)
            value_states = self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            
            # Fused RMSNorm + Transpose + RoPE -> returns [B, H, L, D]
            query_states = triton_fused_rmsnorm_rope(q_view, self_attn.q_norm.weight, cos, sin, self_attn.q_norm.variance_epsilon)
            key_states = triton_fused_rmsnorm_rope(k_view, self_attn.k_norm.weight, cos, sin, self_attn.k_norm.variance_epsilon)

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(key_states, value_states, self_attn.layer_idx, cache_kwargs)

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
                        # Bind the new method
                        layer.self_attn.forward = fused_forward.__get__(layer.self_attn, layer.self_attn.__class__)
                        count += 1
        
        if count > 0 and 'triton_fused_rmsnorm_rope' not in self._optimizations_applied:
            self._optimizations_applied.append('triton_fused_rmsnorm_rope')
        print(f"[VLMModel] Patched {count} Attention layers with Fused RMSNorm+RoPE.")

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

    def _optimize_generation_config(self):
        """优化生成配置
        
        当前默认策略：
        - 启用 use_cache
        - 启用 static cache
        - 关闭采样
        """
        print(f"[VLMModel] Optimizing generation config...")
        status = "enabled"

        if hasattr(self._model, 'generation_config'):
            self._model.generation_config.use_cache = True
            self._model.generation_config.do_sample = False
            self._model.generation_config.temperature = None
            self._model.generation_config.top_p = None
            self._model.generation_config.top_k = None
            self._model.generation_config.cache_implementation = "static"

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
        original_generate = self._model.generate

        def fast_generate(*args, **kwargs):
            if kwargs.get("do_sample") is False:
                kwargs.pop("temperature", None)
                kwargs.pop("top_p", None)
                kwargs.pop("top_k", None)
            kwargs.setdefault("use_cache", True)
            if kwargs.get("do_sample") is False and int(kwargs.get("max_new_tokens", 0)) == 128:
                kwargs.setdefault("min_new_tokens", 128)
            if "pixel_values" in kwargs and isinstance(kwargs["pixel_values"], torch.Tensor):
                if not kwargs["pixel_values"].is_contiguous():
                    kwargs["pixel_values"] = kwargs["pixel_values"].contiguous()
            with torch.inference_mode():
                return original_generate(*args, **kwargs)

        self._model.generate = fast_generate
        if 'generate_fastpath' not in self._optimizations_applied:
            self._optimizations_applied.append('generate_fastpath')

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
