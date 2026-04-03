print("Starting")
import time
import os
os.environ["AICAS_PROFILE"] = "manual_kernel_plus_fastpath"
import torch
import evaluation_wrapper
from PIL import Image
print("Imported")

model_path = "/workspace/Qwen3-VL/AICASGC/Qwen3-VL-2B-Instruct"
wrapper = evaluation_wrapper.VLMModel(model_path)
model = wrapper.model
processor = wrapper.processor
print("Model loaded")

image = Image.new('RGB', (224, 224), color = 'red')
inputs = processor.apply_chat_template(
    [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "What is in the image?"}]}],
    tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to("cuda:0")

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=128, min_new_tokens=128)

torch.cuda.synchronize()
t0 = time.time()
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=128, min_new_tokens=128)
torch.cuda.synchronize()
total_time = time.time() - t0

generated_ids = out[0][inputs["input_ids"].shape[1]:]
print(f"Generated tokens: {len(generated_ids)}")
print(f"Total time: {total_time:.4f} s")
print(f"Throughput: {len(generated_ids)/total_time:.2f} tok/s")
