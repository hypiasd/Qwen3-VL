# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the AICAS 2026 Vision-Language Model (VLM) Optimization Competition repository. Participants optimize the Qwen3-VL-2B-Instruct model for better inference performance while maintaining accuracy.

## Key Files

- **`AICASGC/evaluation_wrapper.py`** - The main file to modify. Contains the `VLMModel` class where optimizations are implemented.
- **`AICASGC/benchmark.py`** - Benchmark script for self-testing. DO NOT MODIFY - official evaluation uses a separate system.
- **`AICASGC/requirements.txt`** - Python dependencies.
- **`AICASGC/data/`** - Validation dataset (arrow format).
- **`算法大赛 第二季/AICASGC_Season2_CN.md`** - Season 2 submission guidelines.

## Code Architecture

### VLMModel Class (evaluation_wrapper.py)

The optimization architecture uses modular design with these predefined optimization methods:

1. `_optimize_vision_encoder()` - Vision encoder acceleration for high-resolution images
2. `_optimize_kv_cache()` - KV Cache management to reduce memory fragmentation
3. `_optimize_cross_modal_connector()` - Cross-modal connector optimization
4. `_enable_flash_attention()` - Flash Attention implementation
5. `_apply_quantization()` - Quantization optimization (INT8/FP8)

Optimizations are applied via Monkey Patching in `__init__`. The `VLMModel` must expose these properties:
- `processor` - AutoProcessor instance
- `model` - The underlying model object (benchmark calls `model.generate()`)
- `device` - CUDA device string

### Evaluation Metrics

Final Score = 0.4 × Accuracy + 0.3 × TTFT_Improvement + 0.3 × Throughput_Improvement

- **TTFT (Time To First Token)**: ~80ms baseline
- **Throughput**: ~55 tokens/sec baseline
- **Accuracy**: VQA accuracy on 5000 samples

## Common Commands

### Install Dependencies

```bash
cd AICASGC
pip install -r requirements.txt
```

### Download Model

```bash
mkdir -p Qwen3-VL-2B-Instruct
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download \
  --resume-download \
  Qwen/Qwen3-VL-2B-Instruct \
  --local-dir ./Qwen3-VL-2B-Instruct \
  --local-dir-use-symlinks False
```

### Run Benchmark (Quick Test)

```bash
python benchmark.py \
    --model-path ./Qwen3-VL-2B-Instruct \
    --dataset-path ./data \
    --output result.json \
    --num-samples 100
```

### Run Full Benchmark (5000 samples)

```bash
python benchmark.py \
    --model-path ./Qwen3-VL-2B-Instruct \
    --dataset-path ./data \
    --output result.json \
    --num-samples 5000
```

## Competition Rules & Guidelines

### Allowed Optimizations

- Operator replacement and kernel optimization (Triton, CUDA C++)
- Memory and cache optimization (KV Cache layout, memory access patterns)
- Compilation and graph optimization (torch.compile)
- Attention mechanism optimization (Flash Attention, memory-efficient attention)
- Generation process optimization (decoding strategies, cache management)

### Prohibited

- Modifying `benchmark.py`
- Hardcoding answers or modifying the dataset
- Using external APIs or services
- Changing model architecture or directly modifying weights
- Overfitting to specific evaluation samples
- Environment manipulation (GPU frequency locking, etc.)

## Evaluation Environment

Official evaluation uses:
- **GPU**: NVIDIA A800 80GB PCIe
- **Python**: 3.12.3
- **PyTorch**: 2.8.0+cu128
- **CUDA**: 12.8
- **OS**: Linux (x86_64)

## Submission

Submit a ZIP file containing:
- `evaluation_wrapper.py` - Your optimized `VLMModel`
- `requirements.txt` - Optional (if different from official)
- Custom kernels/extensions (`.so` files pre-compiled for CUDA 12.8)
- Other helper modules

The ZIP should NOT contain `benchmark.py` (it will be ignored).
