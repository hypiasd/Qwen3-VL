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
from transformers import AutoModelForImageTextToText, AutoProcessor


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

        self._enable_torch_compile()
        self._enable_sdpa_attention()
        self._optimize_generation_config()
        self._enable_tensor_float32()

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
        """优化视觉编码器"""
        if 'vision_encoder' not in self._optimizations_applied:
            self._optimizations_applied.append('vision_encoder')

    def _optimize_kv_cache(self):
        """优化KV Cache"""
        self._model.config.use_cache = True
        if 'kv_cache' not in self._optimizations_applied:
            self._optimizations_applied.append('kv_cache')

    def _optimize_cross_modal_connector(self):
        """优化跨模态连接器"""
        if 'cross_modal' not in self._optimizations_applied:
            self._optimizations_applied.append('cross_modal')

    def _enable_flash_attention(self):
        """启用Flash Attention"""
        if 'flash_attention' not in self._optimizations_applied:
            self._optimizations_applied.append('flash_attention')

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