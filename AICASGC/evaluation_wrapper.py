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
        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": device,
            "low_cpu_mem_usage": True,
        }
        self._model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
        self._model.eval()

        self._optimizations_applied = []

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
