"""
GPTQ quantization benchmark: Llama 3.1 8B on a single A100.

Uses a pre-quantized GPTQ model from Hugging Face.
The quantization (Hessian computation + weight rounding) was done offline;
this script loads the result and benchmarks it.

Configuration:
    - 4-bit, group_size=128, desc_act=True
"""

import gc
import json
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_utils import (
    get_gpu_memory_usage,
    get_model_size_on_disk,
    measure_perplexity,
    measure_throughput,
)

# Pre-quantized GPTQ model (4-bit, group_size=128, desc_act=True)
MODEL_ID = "TechxGenus/Meta-Llama-3.1-8B-GPTQ"
MODEL_NAME = "meta-llama/Llama-3.1-8B (GPTQ 4-bit)"
PROMPT = "Explain the theory of general relativity in simple terms."
MAX_NEW_TOKENS = 256
NUM_RUNS = 5


def get_gpu_environment() -> dict:
    """Collect GPU hardware and software environment details."""
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "gpu_total_memory_gb": torch.cuda.get_device_properties(0).total_memory
        / (1024**3),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }


def main() -> None:
    results: dict = {
        "model": MODEL_ID,
        "method": "gptq",
        "config": "4bit_g128_desc_act",
    }

    # GPU environment
    results["gpu_environment"] = get_gpu_environment()
    print("GPU Environment:")
    for k, v in results["gpu_environment"].items():
        print(f"  {k}: {v}")
    print()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    # Load pre-quantized model
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
    )

    # 1. GPU memory
    results["gpu_memory"] = get_gpu_memory_usage()
    print(f"GPU memory: {results['gpu_memory']['max_allocated_gb']:.2f} GB allocated")

    # 2. Throughput
    print(f"Measuring throughput ({NUM_RUNS} runs)...")
    results["throughput"] = measure_throughput(
        model, tokenizer, PROMPT, MAX_NEW_TOKENS, NUM_RUNS
    )
    print(
        f"Throughput: {results['throughput']['median_tokens_per_sec']:.1f} tokens/sec"
    )

    # 3. Perplexity
    print("Measuring perplexity on wikitext-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    results["perplexity"] = measure_perplexity(model, tokenizer, dataset)
    print(f"Perplexity: {results['perplexity']:.2f}")

    # 4. Model size on disk
    model_path = snapshot_download(MODEL_ID, local_files_only=True)
    results["model_size"] = get_model_size_on_disk(model_path)
    print(f"Model size: {results['model_size']['total_size_gb']:.2f} GB")

    # Print summary
    print(f"\n{'='*50}")
    print("GPTQ BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"  Model:             {MODEL_ID}")
    print(f"  GPU:               {results['gpu_environment']['gpu_name']}")
    print(f"  Model size:        {results['model_size']['total_size_gb']:.2f} GB")
    print(f"  GPU mem allocated:  {results['gpu_memory']['max_allocated_gb']:.2f} GB")
    print(f"  Throughput:        {results['throughput']['median_tokens_per_sec']:.1f} tokens/sec")
    print(f"  Perplexity:        {results['perplexity']:.2f}")
    print(f"{'='*50}")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "results_gptq.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()