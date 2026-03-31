"""
AICAS 2026 - Qwen3-VL 优化实现
"""
from typing import Dict
try:
    from PIL import Image
except ImportError:
    class Image:
        pass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor


class VLMModel:
    """优化后的 VLMModel"""

    def __init__(self, model_path: str, device: str = "cuda:0"):
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

        # ========== 应用优化 ==========
        self._enable_torch_compile()
        self._enable_sdpa_attention()
        self._optimize_generation_config()
        self._enable_tensor_float32()
        self._optimize_vision_encoder()
        self._optimize_kv_cache()
        self._optimize_cross_modal_connector()
        self._enable_flash_attention()
        # =================================

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
            lm = None
            if hasattr(self._model.model, 'language_model'):
                lm = self._model.model.language_model
            elif hasattr(self._model.model, 'layers'):
                lm = self._model.model

            if lm and hasattr(lm, 'layers'):
                for layer in lm.layers:
                    if hasattr(layer, 'self_attn'):
                        layer.self_attn._attn = F.scaled_dot_product_attention
                        layer.self_attn.use_flash_attn_2 = False
                        layer.self_attn.use_mem_efficient_attn = True

        if 'sdpa_attention' not in self._optimizations_applied:
            self._optimizations_applied.append('sdpa_attention')
        print(f"[VLMModel] SDPA attention enabled")

    def _optimize_generation_config(self):
        """优化生成配置"""
        print(f"[VLMModel] Optimizing generation config...")

        if hasattr(self._model, 'generation_config'):
            self._model.generation_config.use_cache = True

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

    def _optimize_vision_encoder(self):
        """优化视觉编码器 - 按照README示例替换attention算子"""
        print(f"[VLMModel] Optimizing vision encoder...")

        # 正确的路径: model.model.visual
        if hasattr(self._model, 'model') and hasattr(self._model.model, 'visual'):
            vision_model = self._model.model.visual

            if hasattr(vision_model, 'blocks'):
                for i, block in enumerate(vision_model.blocks):
                    if hasattr(block, 'attn'):
                        original_attn_forward = block.attn.forward

                        def make_optimized_attn(orig_forward):
                            def optimized_attn(hidden_states, *args, **kwargs):
                                return orig_forward(hidden_states, *args, **kwargs)
                            return optimized_attn

                        block.attn.forward = make_optimized_attn(original_attn_forward)
                        print(f"[VLMModel] Block {i} attn patched")

            if hasattr(vision_model, 'patch_embed'):
                original_patch_embed = vision_model.patch_embed.forward

                def patched_patch_embed(x):
                    return original_patch_embed(x)

                vision_model.patch_embed.forward = patched_patch_embed
                print(f"[VLMModel] patch_embed patched")

        if 'vision_encoder' not in self._optimizations_applied:
            self._optimizations_applied.append('vision_encoder')
        print(f"[VLMModel] Vision encoder optimization applied")

    def _optimize_kv_cache(self):
        """优化KV Cache - 按照README设置use_cache"""
        print(f"[VLMModel] Optimizing KV cache...")

        if hasattr(self._model, 'config'):
            self._model.config.use_cache = True
            if hasattr(self._model.config, 'pad_token_id'):
                if self._model.config.pad_token_id is None:
                    self._model.config.pad_token_id = self._model.config.eos_token_id

        if hasattr(self._model, 'model') and hasattr(self._model.model, 'language_model'):
            lm = self._model.model.language_model
            if hasattr(lm, 'layers'):
                for layer in lm.layers:
                    if hasattr(layer, 'self_attn'):
                        layer.self_attn.past_key_value = None

        if 'kv_cache' not in self._optimizations_applied:
            self._optimizations_applied.append('kv_cache')
        print(f"[VLMModel] KV cache optimization applied")

    def _optimize_cross_modal_connector(self):
        """优化跨模态连接器"""
        print(f"[VLMModel] Optimizing cross-modal connector...")

        if hasattr(self._model, 'model') and hasattr(self._model.model, 'connector'):
            connector = self._model.model.connector

            for name, module in connector.named_modules():
                if isinstance(module, nn.Linear):
                    module.float()
                    for param in module.parameters():
                        param.data = param.data.contiguous()

            print(f"[VLMModel] Connector modules optimized")

        if 'cross_modal_connector' not in self._optimizations_applied:
            self._optimizations_applied.append('cross_modal_connector')
        print(f"[VLMModel] Cross-modal connector optimization applied")

    def _enable_flash_attention(self):
        """启用Flash Attention - 按照README使用PyTorch backend"""
        print(f"[VLMModel] Enabling Flash Attention...")

        import torch.backends.cuda as cuda_backends
        cuda_backends.enable_flash_sdp(True)
        cuda_backends.enable_mem_efficient_sdp(True)
        cuda_backends.enable_math_sdp(False)

        if hasattr(self._model, 'model') and hasattr(self._model.model, 'language_model'):
            lm = self._model.model.language_model
            if hasattr(lm, 'layers'):
                for layer in lm.layers:
                    if hasattr(layer, 'self_attn'):
                        if hasattr(layer.self_attn, 'set_flash_attention'):
                            layer.self_attn.set_flash_attention(True)

        if 'flash_attention' not in self._optimizations_applied:
            self._optimizations_applied.append('flash_attention')
        print(f"[VLMModel] Flash Attention enabled")

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