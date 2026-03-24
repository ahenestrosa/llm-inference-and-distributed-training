"""
Baseline benchmark: Llama 3.1 8B in BF16 on a single A100.

Measures:
    - GPU memory after loading
    - Tokens/second (median of 5 runs)
    - Perplexity on wikitext-2-raw-v1 (first 1000 tokens)
    - Model size on disk
"""

import json

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

MODEL_ID = "meta-llama/Llama-3.1-8B"
PROMPT = "Explain the theory of general relativity in simple terms."
MAX_NEW_TOKENS = 256
NUM_RUNS = 5


def get_gpu_environment() -> dict:
    """Collect GPU hardware and software environment details."""
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "gpu_total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }


def main() -> None:
    results: dict = {"model": MODEL_ID, "dtype": "bfloat16"}

    # GPU environment
    results["gpu_environment"] = get_gpu_environment()
    print("GPU Environment:")
    for k, v in results["gpu_environment"].items():
        print(f"  {k}: {v}")
    print()

    # Reset peak memory stats before loading
    torch.cuda.reset_peak_memory_stats()

    # Load model and tokenizer
    print(f"Loading {MODEL_ID} in BF16...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
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
    print("\n" + "=" * 50)
    print("BF16 BASELINE RESULTS")
    print("=" * 50)
    print(f"  Model:            {MODEL_ID}")
    print(f"  GPU:              {results['gpu_environment']['gpu_name']}")
    print(f"  CUDA:             {results['gpu_environment']['cuda_version']}")
    print(f"  PyTorch:          {results['gpu_environment']['torch_version']}")
    print(f"  Model size:       {results['model_size']['total_size_gb']:.2f} GB")
    print(f"  GPU mem allocated: {results['gpu_memory']['max_allocated_gb']:.2f} GB")
    print(f"  Throughput:       {results['throughput']['median_tokens_per_sec']:.1f} tokens/sec")
    print(f"  Perplexity:       {results['perplexity']:.2f}")
    print("=" * 50)

    # Save results
    output_path = "results_baseline_bf16.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
