"""
Shared benchmarking utilities for quantization experiments.
Reusable across all quantization variants (BF16 baseline, GPTQ, AWQ, etc.).
"""

import os
import time
from pathlib import Path
from statistics import median

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from datasets import Dataset


def measure_throughput(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 256,
    num_runs: int = 5,
) -> dict:
    """Measure generation throughput in tokens/second.

    Steps:
        1. Tokenize the prompt and move input_ids to the model's device.
        2. Run a warmup generation (1 run, not timed) to avoid cold-start effects.
        3. For each of `num_runs`:
            - Record start time
            - Generate with model.generate() using max_new_tokens
            - Record end time
            - Count *only new tokens generated* (output length - input length)
            - Compute tokens/sec for that run
        4. Return the median tokens/sec and the list of all per-run values.

    Returns:
        dict with keys:
            - "median_tokens_per_sec": float
            - "all_runs_tokens_per_sec": list[float]
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Warmup run
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()

    runs = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        new_tokens = output.shape[1] - input_len
        runs.append(new_tokens / elapsed)

    return {
        "median_tokens_per_sec": median(runs),
        "all_runs_tokens_per_sec": runs,
    }


def measure_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    max_length: int = 1024,
    stride: int = 512,
) -> float:
    """Compute perplexity on an eval set using a sliding window approach.

    Steps:
        1. Concatenate all text in the dataset into a single string and tokenize it.
        2. Truncate to `max_tokens` total tokens if needed.
        3. Use a sliding window of size `max_length` with a step of `stride`:
            - For each window, feed tokens into the model to get logits.
            - Only compute loss on the tokens that are *new* in each window
              (i.e., those in the [stride:] portion), to avoid double-counting.
            - Use cross-entropy between the model's logits and the actual next tokens.
        4. Aggregate the negative log-likelihoods and compute perplexity = exp(avg_nll).

    Hint: Set labels to -100 for tokens you don't want to include in the loss.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        dataset: A Hugging Face Dataset with a "text" column.
        max_length: Context window size for each sliding window pass.
        stride: How far to slide the window each step.

    Returns:
        Perplexity as a float.
    """
    device = next(model.parameters()).device
    # Concatenate all text and tokenize
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings["input_ids"].shape[1]

    nlls = []
    num_tokens = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        input_ids = encodings["input_ids"][:, begin:end].to(device)

        # Labels: mask out tokens that overlap with previous window
        labels = input_ids.clone()
        overlap = begin if begin == 0 else max_length - stride
        labels[:, :overlap] = -100

        with torch.no_grad():
            loss = model(input_ids=input_ids, labels=labels).loss

        # Count only non-masked tokens
        valid_tokens = (labels != -100).sum().item()
        nlls.append(loss.item() * valid_tokens)
        num_tokens += valid_tokens

        if end == seq_len:
            break

    return torch.exp(torch.tensor(sum(nlls) / num_tokens)).item()


def get_gpu_memory_usage(device: int = 0) -> dict:
    """Report GPU memory usage stats.

    Use torch.cuda memory tracking functions to get:
        - max_memory_allocated: peak memory used by tensors (this is the main metric)
        - max_memory_reserved: peak memory reserved by the caching allocator

    Remember to reset peak stats *before* loading the model if you want accurate
    measurements, using torch.cuda.reset_peak_memory_stats().

    Args:
        device: CUDA device index.

    Returns:
        dict with keys:
            - "max_allocated_gb": float (in GB)
            - "max_reserved_gb": float (in GB)
    """
    bytes_to_gb = 1 / (1024 ** 3)
    return {
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) * bytes_to_gb,
        "max_reserved_gb": torch.cuda.max_memory_reserved(device) * bytes_to_gb,
    }


def get_model_size_on_disk(model_path: str | Path) -> dict:
    """Calculate total model size on disk from safetensors files.

    Steps:
        1. Walk the model_path directory.
        2. Find all files ending in .safetensors.
        3. Sum their sizes using os.path.getsize().
        4. Also count total number of safetensors files.

    Args:
        model_path: Path to the model directory (e.g., HF cache or local folder).

    Returns:
        dict with keys:
            - "total_size_gb": float (in GB)
            - "num_files": int
    """
    model_path = Path(model_path)
    total_size = 0
    num_files = 0

    for file in model_path.rglob("*.safetensors"):
        total_size += os.path.getsize(file)
        num_files += 1

    return {
        "total_size_gb": total_size / (1024 ** 3),
        "num_files": num_files,
    }
