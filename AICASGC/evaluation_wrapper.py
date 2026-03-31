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
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


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

    def __init__(self, model_path: str, device: str = "cuda:0", use_quantization: bool = False):
        """
        Initialize model and apply optimizations.

        Args:
            model_path: Qwen3-VL-2B-Instruct model path
            device: CUDA device, e.g., "cuda:0"
            use_quantization: 是否使用INT8量化
        """
        self._device = device
        self.model_path = model_path
        self._use_quantization = use_quantization

        print(f"[VLMModel] Loading processor from {model_path}...")
        self._processor = AutoProcessor.from_pretrained(model_path)

        if use_quantization:
            print(f"[VLMModel] Loading model with INT8 quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map=device,
                low_cpu_mem_usage=True
            )
        else:
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

        self._enable_torch_compile()
        self._enable_sdpa_attention()
        self._optimize_generation_config()
        self._enable_tensor_float32()
        
        if use_quantization:
            self._optimizations_applied.append('int8_quantization')

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
        
        优化方向:
        1. 启用Flash Attention for Vision
        2. 优化Patch Embedding
        """
        print(f"[VLMModel] Optimizing vision encoder...")
        
        visual = None
        if hasattr(self._model, 'model') and hasattr(self._model.model, 'visual'):
            visual = self._model.model.visual
        elif hasattr(self._model, 'visual'):
            visual = self._model.visual
        
        if visual is not None:
            torch.backends.cuda.enable_flash_sdp(True)
            
            if hasattr(visual, 'blocks'):
                for block in visual.blocks:
                    if hasattr(block, 'attn'):
                        if hasattr(block.attn, '_attn_implementation'):
                            block.attn._attn_implementation = "flash_attention_2"
        
        if 'vision_encoder' not in self._optimizations_applied:
            self._optimizations_applied.append('vision_encoder')
        print(f"[VLMModel] Vision encoder optimized")

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
        """启用Flash Attention
        
        方法1: 启用PyTorch内置Flash Attention (简单)
        方法2: 自定义Triton Kernel (高级)
        """
        print(f"[VLMModel] Enabling Flash Attention...")
        
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        
        if 'flash_attention' not in self._optimizations_applied:
            self._optimizations_applied.append('flash_attention')
        print(f"[VLMModel] Flash Attention enabled")

    def _apply_quantization(self):
        """应用量化优化
        
        优化方向:
        1. INT8动态量化
        2. FP8量化 (需要硬件支持)
        3. KV Cache量化
        
        注意: 
        - 量化在RTX 3090上会降低性能（INT8矩阵乘法不如FP16）
        - 量化主要用于减少显存占用，而非提升速度
        - 如需使用，请在模型加载时设置use_quantization=True
        """
        print(f"[VLMModel] Quantization skipped - FP16 is optimal for RTX 3090")
        print(f"[VLMModel] Use use_quantization=True at init for INT8 (reduces VRAM, slower)")
        
        if 'quantization' not in self._optimizations_applied:
            self._optimizations_applied.append('quantization_skipped')

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