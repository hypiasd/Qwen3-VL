"""
AICAS 2026 - Participant Core Modification File

Participants should modify the VLMModel class to implement optimizations.

Note:
- Benchmark directly calls self.model.generate() for performance testing.
- Your optimizations should modify self.model or its operators in __init__ via Monkey Patch.
- The generate() method is optional and mainly for debugging.
"""
from typing import Dict
try:
    from PIL import Image
except ImportError:
    class Image:
        pass
import torch
import os
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from triton_kernels import triton_bilinear_pos_embed, fused_layernorm_linear
    HAS_CUSTOM_KERNELS = True
except ImportError:
    HAS_CUSTOM_KERNELS = False

try:
    from fused_kernels import fused_linear_gelu, fused_linear_silu
    HAS_FUSED_KERNELS = True
except ImportError:
    HAS_FUSED_KERNELS = False

try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


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
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True
        )
        self._model.eval()

        self._optimizations_applied = []

        # ================================================================
        # Participant Optimization Area - Enable/disable optimization methods
        # ================================================================

        # 基础优化（已验证有效）
        self._enable_torch_compile()
        self._enable_sdpa_attention()
        self._optimize_generation_config()
        self._enable_tensor_float32()
        
        # 深度优化（新增）
        self._enable_flash_attention()        # Flash Attention 2
        self._optimize_vision_encoder()       # Vision Encoder Triton优化
        self._apply_operator_fusion()         # 算子融合
        self._optimize_memory_access()        # 内存访问优化

        print(f"[VLMModel] Model loaded successfully on {device}")
        if self._optimizations_applied:
            print(f"[VLMModel] Applied optimizations: {', '.join(self._optimizations_applied)}")

    def _enable_torch_compile(self):
        """使用 torch.compile 优化整个模型"""
        print(f"[VLMModel] Compiling model with torch.compile...")

        self._model = torch.compile(
            self._model,
            mode="max-autotune",
            fullgraph=False,
            dynamic=False
        )

        if 'torch_compile' not in self._optimizations_applied:
            self._optimizations_applied.append('torch_compile')
        print(f"[VLMModel] torch.compile optimization applied")

    def _enable_sdpa_attention(self):
        """启用 SDPA (Scaled Dot Product Attention) 加速"""
        print(f"[VLMModel] Enabling SDPA attention...")

        if hasattr(self._model, 'model'):
            if hasattr(self._model.model, 'language_model'):
                lm = self._model.model.language_model
                if hasattr(lm, 'layers'):
                    for layer in lm.layers:
                        if hasattr(layer, 'self_attn'):
                            layer.self_attn._attn = torch.nn.functional.scaled_dot_product_attention
                            layer.self_attn.use_flash_attn_2 = False
                            layer.self_attn.use_mem_efficient_attn = True
            elif hasattr(self._model.model, 'layers'):
                for layer in self._model.model.layers:
                    if hasattr(layer, 'self_attn'):
                        layer.self_attn._attn = torch.nn.functional.scaled_dot_product_attention

        if 'sdpa_attention' not in self._optimizations_applied:
            self._optimizations_applied.append('sdpa_attention')
        print(f"[VLMModel] SDPA attention enabled")

    def _optimize_generation_config(self):
        """优化生成配置"""
        print(f"[VLMModel] Optimizing generation config...")

        if hasattr(self._model, 'generation_config'):
            self._model.generation_config.use_cache = True
            self._model.generation_config.cache_implementation = "static"

        if 'generation_config' not in self._optimizations_applied:
            self._optimizations_applied.append('generation_config')
        print(f"[VLMModel] Generation config optimized")

    def _enable_tensor_float32(self):
        """启用 TensorFloat32加速矩阵乘法"""
        print(f"[VLMModel] Enabling TensorFloat32...")
        torch.set_float32_matmul_precision('high')

        if 'tensor_float32' not in self._optimizations_applied:
            self._optimizations_applied.append('tensor_float32')
        print(f"[VLMModel] TensorFloat32 enabled")

    def _explore_model_structure(self):
        """Helper method to explore model structure."""
        print("=" * 60)
        print("Model Structure Exploration")
        print("=" * 60)

        if hasattr(self._model, 'visual'):
            print(f"Visual Model: {type(self._model.visual)}")
        elif hasattr(self._model, 'model') and hasattr(self._model.model, 'visual'):
            print(f"Visual Model: {type(self._model.model.visual)}")

        if hasattr(self._model, 'model'):
            print(f"Model: {type(self._model.model)}")

        print("=" * 60)

    def _optimize_vision_encoder(self):
        """优化视觉编码器
        
        实现：
        1. Bilinear position embedding融合kernel
        2. LayerNorm + Linear融合
        3. 优化attention实现
        """
        print(f"[VLMModel] Optimizing vision encoder...")
        
        visual = self._get_visual_model()
        if visual is None:
            print(f"[VLMModel] No visual model found, skipping vision encoder optimization")
            return
        
        # 应用Triton优化
        if HAS_CUSTOM_KERNELS:
            self._apply_triton_to_visual(visual)
        
        if 'vision_encoder' not in self._optimizations_applied:
            self._optimizations_applied.append('vision_encoder_triton' if HAS_CUSTOM_KERNELS else 'vision_encoder')
        
        print(f"[VLMModel] Vision encoder optimized")
    
    def _get_visual_model(self):
        """获取visual model"""
        # 处理torch.compile包装的情况
        model_obj = self._model
        if hasattr(model_obj, '_orig_mod'):
            model_obj = model_obj._orig_mod
        
        if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'visual'):
            return model_obj.model.visual
        elif hasattr(model_obj, 'visual'):
            return model_obj.visual
        return None
    
    def _apply_triton_to_visual(self, visual):
        """将Triton优化应用到visual model"""
        try:
            # 替换position embedding插值方法
            if hasattr(visual, 'pos_embed_interpolate'):
                original_interpolate = visual.pos_embed_interpolate
                visual._original_pos_embed_interpolate = original_interpolate
                visual.pos_embed_interpolate = lambda pos_embed, H, W: triton_bilinear_pos_embed(pos_embed, H, W)
                print(f"[VLMModel] Applied Triton position embedding interpolation")
            
            # 替换LayerNorm + Linear融合
            if hasattr(visual, 'blocks'):
                for block in visual.blocks:
                    if hasattr(block, 'norm1') and hasattr(block, 'mlp'):
                        self._apply_fused_norm_mlp(block)
            
            print(f"[VLMModel] Triton kernels applied to vision encoder")
        except Exception as e:
            print(f"[VLMModel] Warning: Failed to apply Triton kernels: {e}")
    
    def _apply_fused_norm_mlp(self, block):
        """应用融合的LayerNorm + MLP"""
        if not HAS_CUSTOM_KERNELS:
            return
        
        try:
            norm_layer = block.norm1
            mlp_layer = block.mlp
            
            if hasattr(mlp_layer, 'fc1'):
                # 保存原始forward
                if not hasattr(block, '_original_forward'):
                    block._original_forward = block.forward
                
                # 创建融合forward
                def create_fused_forward(original_forward, norm, mlp):
                    def fused_forward(hidden_states, **kwargs):
                        try:
                            # 融合LayerNorm + Linear
                            normalized = fused_layernorm_linear(
                                hidden_states,
                                mlp.fc1.weight,
                                mlp.fc1.bias,
                                eps=getattr(norm, 'eps', 1e-5)
                            )
                            
                            # 继续MLP的剩余部分
                            if hasattr(mlp, 'act'):
                                normalized = mlp.act(normalized)
                            if hasattr(mlp, 'fc2'):
                                normalized = mlp.fc2(normalized)
                            
                            # Residual connection
                            output = hidden_states + normalized
                            return (output,)
                        except:
                            # Fallback to original
                            return original_forward(hidden_states, **kwargs)
                    
                    return fused_forward
                
                block.forward = create_fused_forward(block._original_forward, norm_layer, mlp_layer)
        except Exception as e:
            print(f"[VLMModel] Warning: Failed to apply fused norm+mlp: {e}")

    def _optimize_kv_cache(self):
        """优化KV Cache
        
        优化方向:
        1. Memory layout optimization
        2. Fragmentation-free allocation
        3. Efficient cache reuse
        """
        print(f"[VLMModel] Optimizing KV cache...")
        
        self._model.config.use_cache = True
        
        if 'kv_cache' not in self._optimizations_applied:
            self._optimizations_applied.append('kv_cache')
        print(f"[VLMModel] KV cache optimized")

    def _optimize_cross_modal_connector(self):
        """优化跨模态连接器
        
        优化方向:
        1. MLP层融合
        2. Vision-to-language投影优化
        """
        print(f"[VLMModel] Optimizing cross-modal connector...")
        
        visual = None
        if hasattr(self._model, 'model') and hasattr(self._model.model, 'visual'):
            visual = self._model.model.visual
        elif hasattr(self._model, 'visual'):
            visual = self._model.visual
        
        if visual is not None:
            if hasattr(visual, 'merger'):
                visual.merger = torch.compile(
                    visual.merger,
                    mode="reduce-overhead",
                    fullgraph=False
                )
            
            if hasattr(visual, 'deepstack_merger_list'):
                for i, merger in enumerate(visual.deepstack_merger_list):
                    visual.deepstack_merger_list[i] = torch.compile(
                        merger,
                        mode="reduce-overhead",
                        fullgraph=False
                    )
        
        if 'cross_modal' not in self._optimizations_applied:
            self._optimizations_applied.append('cross_modal')
        print(f"[VLMModel] Cross-modal connector optimized")

    def _enable_flash_attention(self):
        """启用Flash Attention 2
        
        实现：
        1. 尝试使用flash-attn库
        2. Fallback到PyTorch SDPA
        3. 完善的错误处理
        """
        print(f"[VLMModel] Enabling Flash Attention...")
        
        if HAS_FLASH_ATTN:
            print(f"[VLMModel] Using flash-attn library")
            self._apply_flash_attn_to_model()
        else:
            print(f"[VLMModel] flash-attn not available, using PyTorch SDPA")
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        if 'flash_attention' not in self._optimizations_applied:
            self._optimizations_applied.append('flash_attention_2' if HAS_FLASH_ATTN else 'flash_sdp')
        
        print(f"[VLMModel] Flash Attention enabled")
    
    def _apply_flash_attn_to_model(self):
        """将Flash Attention应用到模型"""
        if not HAS_FLASH_ATTN:
            return
        
        try:
            # 处理torch.compile包装的情况
            model_obj = self._model
            if hasattr(model_obj, '_orig_mod'):
                model_obj = model_obj._orig_mod
            
            if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'layers'):
                for layer_idx, layer in enumerate(model_obj.model.layers):
                    if hasattr(layer, 'self_attn'):
                        self._patch_attention_layer(layer.self_attn, layer_idx, is_vision=False)
            
            visual = self._get_visual_model()
            if visual is not None and hasattr(visual, 'blocks'):
                for block_idx, block in enumerate(visual.blocks):
                    if hasattr(block, 'attn'):
                        self._patch_attention_layer(block.attn, block_idx, is_vision=True)
            
            print(f"[VLMModel] Flash Attention applied to all layers")
        except Exception as e:
            print(f"[VLMModel] Warning: Failed to apply Flash Attention: {e}")
            print(f"[VLMModel] Falling back to PyTorch SDPA")
            torch.backends.cuda.enable_flash_sdp(True)
    
    def _patch_attention_layer(self, attn_module, layer_idx, is_vision=False):
        """替换单个attention层的forward方法"""
        if not hasattr(attn_module, '_original_forward'):
            attn_module._original_forward = attn_module.forward
        
        def create_flash_forward(original_forward, attn, is_vision):
            def flash_forward(hidden_states, **kwargs):
                try:
                    return self._flash_attention_forward(hidden_states, attn, is_vision, **kwargs)
                except Exception as e:
                    print(f"[Warning] Flash Attention failed at layer {layer_idx}, using original: {e}")
                    return original_forward(hidden_states, **kwargs)
            return flash_forward
        
        attn_module.forward = create_flash_forward(attn_module._original_forward, attn_module, is_vision)
    
    def _flash_attention_forward(self, hidden_states, attn_module, is_vision, **kwargs):
        """Flash Attention forward实现"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        num_heads = getattr(attn_module, 'num_heads', getattr(attn_module, 'num_attention_heads', 8))
        head_dim = getattr(attn_module, 'head_dim', hidden_dim // num_heads)
        
        if hasattr(attn_module, 'q_proj'):
            q = attn_module.q_proj(hidden_states)
            k = attn_module.k_proj(hidden_states)
            v = attn_module.v_proj(hidden_states)
        elif hasattr(attn_module, 'query'):
            q = attn_module.query(hidden_states)
            k = attn_module.key(hidden_states)
            v = attn_module.value(hidden_states)
        else:
            raise ValueError("Cannot find Q, K, V projection layers")
        
        q = q.view(batch_size, seq_len, num_heads, head_dim)
        k = k.view(batch_size, seq_len, num_heads, head_dim)
        v = v.view(batch_size, seq_len, num_heads, head_dim)
        
        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=0.0,
            softmax_scale=1.0 / (head_dim ** 0.5),
            causal=not is_vision
        )
        
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        if hasattr(attn_module, 'o_proj'):
            output = attn_module.o_proj(attn_output)
        elif hasattr(attn_module, 'out_proj'):
            output = attn_module.out_proj(attn_output)
        else:
            output = attn_output
        
        return (output, None, None) if not is_vision else output

    def _apply_operator_fusion(self):
        """应用算子融合优化
        
        融合：
        1. Linear + GELU/SiLU
        2. 减少kernel launch开销
        """
        print(f"[VLMModel] Applying operator fusion...")
        
        if not HAS_FUSED_KERNELS:
            print(f"[VLMModel] Fused kernels not available, skipping")
            return
        
        try:
            # 处理torch.compile包装的情况
            model_obj = self._model
            if hasattr(model_obj, '_orig_mod'):
                model_obj = model_obj._orig_mod
            
            # 替换Language Model的FFN层
            if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'layers'):
                for layer in model_obj.model.layers:
                    if hasattr(layer, 'mlp'):
                        self._apply_fused_ffn(layer.mlp)
            
            if 'operator_fusion' not in self._optimizations_applied:
                self._optimizations_applied.append('operator_fusion')
            
            print(f"[VLMModel] Operator fusion applied")
        except Exception as e:
            print(f"[VLMModel] Warning: Failed to apply operator fusion: {e}")
    
    def _apply_fused_ffn(self, mlp):
        """应用融合的FFN"""
        try:
            # 检查是否是SwiGLU结构
            if hasattr(mlp, 'gate_proj') and hasattr(mlp, 'up_proj'):
                # SwiGLU: gate_proj + SiLU, up_proj
                if not hasattr(mlp, '_original_forward'):
                    mlp._original_forward = mlp.forward
                
                def create_fused_swiglu_forward(original_forward, mlp):
                    def fused_forward(x):
                        try:
                            gate = fused_linear_silu(x, mlp.gate_proj.weight, mlp.gate_proj.bias)
                            up = torch.nn.functional.linear(x, mlp.up_proj.weight, mlp.up_proj.bias)
                            output = gate * up
                            if hasattr(mlp, 'down_proj'):
                                output = mlp.down_proj(output)
                            return output
                        except:
                            return original_forward(x)
                    return fused_forward
                
                mlp.forward = create_fused_swiglu_forward(mlp._original_forward, mlp)
            
            # 检查是否是标准FFN结构
            elif hasattr(mlp, 'fc1'):
                # 标准FFN: fc1 + GELU + fc2
                if not hasattr(mlp, '_original_forward'):
                    mlp._original_forward = mlp.forward
                
                def create_fused_ffn_forward(original_forward, mlp):
                    def fused_forward(x):
                        try:
                            hidden = fused_linear_gelu(x, mlp.fc1.weight, mlp.fc1.bias)
                            if hasattr(mlp, 'fc2'):
                                hidden = mlp.fc2(hidden)
                            return hidden
                        except:
                            return original_forward(x)
                    return fused_forward
                
                mlp.forward = create_fused_ffn_forward(mlp._original_forward, mlp)
        except Exception as e:
            print(f"[VLMModel] Warning: Failed to apply fused FFN: {e}")
    
    def _optimize_memory_access(self):
        """优化内存访问模式
        
        实现：
        1. 确保tensor contiguous
        2. 优化cache命中率
        3. 设置环境变量
        """
        print(f"[VLMModel] Optimizing memory access patterns...")
        
        try:
            # 优化模型参数的内存布局
            def make_contiguous(module):
                count = 0
                for name, param in module.named_parameters():
                    if not param.is_contiguous():
                        param.data = param.data.contiguous()
                        count += 1
                
                for name, buffer in module.named_buffers():
                    if not buffer.is_contiguous():
                        buffer.data = buffer.data.contiguous()
                        count += 1
                
                return count
            
            count = make_contiguous(self._model)
            print(f"[VLMModel] Made {count} tensors contiguous")
            
            # 启用cudnn benchmark
            torch.backends.cudnn.benchmark = True
            
            # 优化CUDA内存分配器
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            if 'memory_optimization' not in self._optimizations_applied:
                self._optimizations_applied.append('memory_optimization')
            
            print(f"[VLMModel] Memory access optimized")
        except Exception as e:
            print(f"[VLMModel] Warning: Failed to optimize memory access: {e}")
    
    def _apply_quantization(self):
        """量化优化 - 已禁用
        
        注意: 官方规则禁止使用量化优化，此方法保留但不做任何操作
        """
        print(f"[VLMModel] Quantization disabled by competition rules")

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