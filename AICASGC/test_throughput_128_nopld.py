import time
import os
os.environ["AICAS_PROFILE"] = "manual_kernel_plus_fastpath"
import torch
import evaluation_wrapper
from PIL import Image

model_path = "/workspace/Qwen3-VL/AICASGC/Qwen3-VL-2B-Instruct"
wrapper = evaluation_wrapper.VLMModel(model_path)
wrapper.model.generation_config.prompt_lookup_num_tokens = 0
model = wrapper.model
processor = wrapper.processor

# disable our PLD
# Oh wait, our PLD is hardcoded in `generate_fastpath_graph`. 
# I will temporarily disable it in evaluation_wrapper.py if I want to test.
