"""
bitsandbytes quantization benchmark: Llama 3.1 8B on a single A100.

Configurations:
    1. 8-bit (LLM.int8()) — outlier features in FP16, rest in INT8
    2. 4-bit FP4 — standard 4-bit quantization
    3. 4-bit NF4 — normally-distributed quantization type
    4. 4-bit NF4 + double quantization — quantizes the quantization constants
"""

import gc
import json
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import PreTrainedTokenizerBase

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

CONFIGS: dict[str, BitsAndBytesConfig] = {
    "int8": BitsAndBytesConfig(load_in_8bit=True),
    "fp4": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
    ),
    "nf4": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
    ),
    "nf4_double_quant": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ),
}


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


def benchmark_config(
    config_name: str,
    bnb_config: BitsAndBytesConfig,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    model_path: str,
) -> dict:
    """Load model with a given bnb config and run all benchmarks."""
    print(f"\n{'='*50}")
    print(f"Benchmarking: {config_name}")
    print(f"{'='*50}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    # Load model
    print(f"  Loading {MODEL_ID} with {config_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
    )

    result: dict = {"config": config_name}

    # 1. GPU memory
    result["gpu_memory"] = get_gpu_memory_usage()
    print(f"  GPU memory: {result['gpu_memory']['max_allocated_gb']:.2f} GB")

    # 2. Throughput
    print(f"  Measuring throughput ({NUM_RUNS} runs)...")
    result["throughput"] = measure_throughput(
        model, tokenizer, PROMPT, MAX_NEW_TOKENS, NUM_RUNS
    )
    print(f"  Throughput: {result['throughput']['median_tokens_per_sec']:.1f} tokens/sec")

    # 3. Perplexity
    print("  Measuring perplexity on wikitext-2...")
    result["perplexity"] = measure_perplexity(model, tokenizer, dataset)
    print(f"  Perplexity: {result['perplexity']:.2f}")

    # 4. Model size on disk (same for all — bnb quantizes at load time)
    result["model_size"] = get_model_size_on_disk(model_path)
    print(f"  Model size on disk: {result['model_size']['total_size_gb']:.2f} GB")

    # Free GPU memory before next config
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main() -> None:
    all_results: dict = {"model": MODEL_ID, "method": "bitsandbytes"}

    # GPU environment
    all_results["gpu_environment"] = get_gpu_environment()
    print("GPU Environment:")
    for k, v in all_results["gpu_environment"].items():
        print(f"  {k}: {v}")

    # Shared resources
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    model_path = snapshot_download(MODEL_ID, local_files_only=True)

    # Run benchmarks
    all_results["configs"] = {}
    for config_name, bnb_config in CONFIGS.items():
        result = benchmark_config(
            config_name, bnb_config, tokenizer, dataset, model_path
        )
        all_results["configs"][config_name] = result

    # Print comparison table
    print(f"\n{'='*70}")
    print("BITSANDBYTES QUANTIZATION RESULTS")
    print(f"{'='*70}")
    print(f"  {'Config':<20} {'Memory (GB)':>12} {'Tokens/sec':>12} {'Perplexity':>12}")
    print(f"  {'-'*56}")
    for name, r in all_results["configs"].items():
        mem = r["gpu_memory"]["max_allocated_gb"]
        tps = r["throughput"]["median_tokens_per_sec"]
        ppl = r["perplexity"]
        print(f"  {name:<20} {mem:>12.2f} {tps:>12.1f} {ppl:>12.2f}")
    print(f"{'='*70}")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "results_bnb.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
