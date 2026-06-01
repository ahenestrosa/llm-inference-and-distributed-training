"""
Attention-implementation benchmark: Llama 3.1 8B in BF16 on a single A100.

Compares three attention backends across a sweep of sequence lengths:

    - eager:              standard PyTorch attention. Materializes the full
                          [b, h, n, n] scores matrix in HBM. O(n^2) memory.
    - sdpa:               torch.nn.functional.scaled_dot_product_attention.
                          PyTorch picks a backend at runtime — on Ampere+ with
                          FP16/BF16 it dispatches to a FlashAttention kernel,
                          falling back to a memory-efficient implementation
                          otherwise.
    - flash_attention_2:  the FA2 kernel (Dao, 2023). Tiles Q/K/V into SRAM,
                          runs online softmax block-by-block, and never writes
                          the n x n scores back to HBM. ~2x over FA1, parallel
                          over batch x heads x Q row blocks. See
                          knowledge-base/attention/flash-attention-2/ for the
                          underlying algorithm.

At each sequence length we measure:
    1. Prefill forward-pass latency and the peak GPU memory it reaches.
    2. Decode tokens/sec for a short generation continuing from the prefill,
       plus the peak memory of the full prefill + decode run.

batch_size is fixed at 1 to isolate per-sequence behavior — at b=1 the eager
attention scores matrix is the dominant memory cost and the gap to FA2 is
sharpest.

Out-of-memory failures are recorded per (implementation, seq_len) rather than
killing the run; eager attention is expected to OOM well before sdpa/FA2 do.
"""

import gc
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

MODEL_ID = "meta-llama/Llama-3.1-8B"
SEQUENCE_LENGTHS: list[int] = [512, 1024, 2048, 4096, 8192, 16384]
ATTENTION_IMPLEMENTATIONS: list[str] = ["eager", "sdpa", "flash_attention_2"]
GENERATE_NEW_TOKENS = 64
NUM_RUNS = 3
WARMUP_RUNS = 1
BATCH_SIZE = 1


@dataclass
class GpuEnvironment:
    gpu_name: str
    gpu_count: int
    gpu_total_memory_gb: float
    cuda_version: str | None
    torch_version: str


@dataclass
class MemoryStats:
    max_allocated_gb: float
    max_reserved_gb: float

    @classmethod
    def snapshot(cls, device: int = 0) -> "MemoryStats":
        bytes_to_gb = 1 / (1024**3)
        return cls(
            max_allocated_gb=torch.cuda.max_memory_allocated(device) * bytes_to_gb,
            max_reserved_gb=torch.cuda.max_memory_reserved(device) * bytes_to_gb,
        )


@dataclass
class PrefillMeasurement:
    seq_len: int
    median_time_sec: float
    all_times_sec: list[float]
    peak_memory: MemoryStats | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class GenerationMeasurement:
    seq_len: int
    new_tokens_target: int
    median_tokens_per_sec: float
    all_runs_tokens_per_sec: list[float]
    peak_memory: MemoryStats | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ImplementationResults:
    implementation: str
    available: bool
    load_memory: MemoryStats | None
    prefill: list[PrefillMeasurement] = field(default_factory=list)
    generation: list[GenerationMeasurement] = field(default_factory=list)
    load_error: str | None = None


@dataclass
class BenchmarkResults:
    model: str
    dtype: str
    batch_size: int
    sequence_lengths: list[int]
    generate_new_tokens: int
    num_runs: int
    gpu_environment: GpuEnvironment
    implementations: list[ImplementationResults] = field(default_factory=list)


def get_gpu_environment() -> GpuEnvironment:
    return GpuEnvironment(
        gpu_name=torch.cuda.get_device_name(0),
        gpu_count=torch.cuda.device_count(),
        gpu_total_memory_gb=torch.cuda.get_device_properties(0).total_memory / (1024**3),
        cuda_version=torch.version.cuda,
        torch_version=torch.__version__,
    )


def make_random_input(
    tokenizer: PreTrainedTokenizerBase, seq_len: int, device: torch.device
) -> torch.Tensor:
    """Random in-vocab token ids — sufficient for benchmarking shape/compute."""
    return torch.randint(
        low=0,
        high=tokenizer.vocab_size,
        size=(BATCH_SIZE, seq_len),
        device=device,
        dtype=torch.long,
    )


def free_cuda_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def load_model(implementation: str) -> tuple[PreTrainedModel | None, MemoryStats | None, str | None]:
    """Load Llama 3.1 8B in BF16 with the requested attention backend.

    Returns (model, load_memory, error). On failure (e.g., flash-attn not
    installed, unsupported GPU) the first two are None and `error` carries
    the exception message.
    """
    free_cuda_memory()
    torch.cuda.reset_peak_memory_stats()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=implementation,
        )
    except (ImportError, ValueError, RuntimeError) as e:
        return None, None, f"{type(e).__name__}: {e}"

    model.eval()
    return model, MemoryStats.snapshot(), None


def measure_prefill(
    model: PreTrainedModel, input_ids: torch.Tensor
) -> PrefillMeasurement:
    """Time a single forward pass with use_cache=True (true prefill)."""
    seq_len = input_ids.shape[1]

    try:
        for _ in range(WARMUP_RUNS):
            with torch.inference_mode():
                _ = model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        times: list[float] = []
        for _ in range(NUM_RUNS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                _ = model(input_ids=input_ids, use_cache=True)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        return PrefillMeasurement(
            seq_len=seq_len,
            median_time_sec=median(times),
            all_times_sec=times,
            peak_memory=MemoryStats.snapshot(),
        )
    except torch.cuda.OutOfMemoryError as e:
        free_cuda_memory()
        return PrefillMeasurement(
            seq_len=seq_len,
            median_time_sec=float("nan"),
            all_times_sec=[],
            peak_memory=None,
            error=f"OutOfMemoryError: {e}",
        )


def measure_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    input_ids: torch.Tensor,
) -> GenerationMeasurement:
    """Generate `GENERATE_NEW_TOKENS` from `input_ids` and report decode speed.

    Counts only tokens produced after the prefill (output_len - input_len) so
    the figure is purely the per-step decode throughput — though the per-step
    cost itself depends on the prefill length via KV-cache reads.
    """
    seq_len = input_ids.shape[1]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    try:
        with torch.inference_mode():
            _ = model.generate(  # pyright: ignore[reportCallIssue]
                input_ids=input_ids,
                max_new_tokens=GENERATE_NEW_TOKENS,
                do_sample=False,
                pad_token_id=pad_id,
            )
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        runs: list[float] = []
        for _ in range(NUM_RUNS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(  # pyright: ignore[reportCallIssue]
                    input_ids=input_ids,
                    max_new_tokens=GENERATE_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            new_tokens = output.shape[1] - seq_len
            runs.append(new_tokens / elapsed)

        return GenerationMeasurement(
            seq_len=seq_len,
            new_tokens_target=GENERATE_NEW_TOKENS,
            median_tokens_per_sec=median(runs),
            all_runs_tokens_per_sec=runs,
            peak_memory=MemoryStats.snapshot(),
        )
    except torch.cuda.OutOfMemoryError as e:
        free_cuda_memory()
        return GenerationMeasurement(
            seq_len=seq_len,
            new_tokens_target=GENERATE_NEW_TOKENS,
            median_tokens_per_sec=float("nan"),
            all_runs_tokens_per_sec=[],
            peak_memory=None,
            error=f"OutOfMemoryError: {e}",
        )


def benchmark_implementation(
    implementation: str, tokenizer: PreTrainedTokenizerBase
) -> ImplementationResults:
    print(f"\n{'=' * 70}")
    print(f"Benchmarking attn_implementation = {implementation!r}")
    print(f"{'=' * 70}")

    model, load_memory, load_error = load_model(implementation)
    if model is None:
        print(f"  Skipping: {load_error}")
        return ImplementationResults(
            implementation=implementation,
            available=False,
            load_memory=None,
            load_error=load_error,
        )

    assert load_memory is not None
    print(f"  Loaded. Resident memory: {load_memory.max_allocated_gb:.2f} GB")

    device = next(model.parameters()).device
    result = ImplementationResults(
        implementation=implementation,
        available=True,
        load_memory=load_memory,
    )

    for seq_len in SEQUENCE_LENGTHS:
        print(f"\n  seq_len = {seq_len}")
        input_ids = make_random_input(tokenizer, seq_len, device)

        prefill = measure_prefill(model, input_ids)
        result.prefill.append(prefill)
        if prefill.ok and prefill.peak_memory is not None:
            print(
                f"    prefill:    {prefill.median_time_sec * 1000:8.1f} ms"
                f"   peak {prefill.peak_memory.max_allocated_gb:5.2f} GB"
            )
        else:
            print(f"    prefill:    FAILED ({prefill.error})")

        generation = measure_generation(model, tokenizer, input_ids)
        result.generation.append(generation)
        if generation.ok and generation.peak_memory is not None:
            print(
                f"    decode:     {generation.median_tokens_per_sec:8.1f} tok/s"
                f"   peak {generation.peak_memory.max_allocated_gb:5.2f} GB"
            )
        else:
            print(f"    decode:     FAILED ({generation.error})")

    del model
    free_cuda_memory()
    return result


def print_summary(results: BenchmarkResults) -> None:
    print(f"\n{'=' * 90}")
    print("ATTENTION IMPLEMENTATION COMPARISON")
    print(f"{'=' * 90}")

    impls = [r for r in results.implementations if r.available]
    if not impls:
        print("  No implementations ran successfully.")
        return

    header = f"  {'seq_len':>8}  " + "  ".join(
        f"{r.implementation:^28}" for r in impls
    )
    print(header)
    sub = f"  {'':>8}  " + "  ".join(
        f"{'prefill (ms)':>12} {'tok/s':>6} {'mem GB':>7}" for _ in impls
    )
    print(sub)
    print(f"  {'-' * (len(header) - 2)}")

    for i, seq_len in enumerate(results.sequence_lengths):
        cells: list[str] = []
        for r in impls:
            prefill = r.prefill[i] if i < len(r.prefill) else None
            generation = r.generation[i] if i < len(r.generation) else None

            if prefill and prefill.ok:
                pf_str = f"{prefill.median_time_sec * 1000:>12.1f}"
                mem_source = generation.peak_memory if (
                    generation and generation.peak_memory
                ) else prefill.peak_memory
                mem_str = f"{mem_source.max_allocated_gb:>7.2f}" if mem_source else f"{'-':>7}"
            else:
                pf_str = f"{'OOM':>12}"
                mem_str = f"{'-':>7}"

            if generation and generation.ok:
                tok_str = f"{generation.median_tokens_per_sec:>6.1f}"
            else:
                tok_str = f"{'OOM':>6}"

            cells.append(f"{pf_str} {tok_str} {mem_str}")
        print(f"  {seq_len:>8}  " + "  ".join(cells))
    print(f"{'=' * 90}")


def save_results(results: BenchmarkResults) -> Path:
    output_path = Path(__file__).parent / "results_attention.json"
    with open(output_path, "w") as f:
        json.dump(asdict(results), f, indent=2)
    return output_path


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    gpu_env = get_gpu_environment()
    print("GPU environment:")
    for k, v in asdict(gpu_env).items():
        print(f"  {k}: {v}")

    results = BenchmarkResults(
        model=MODEL_ID,
        dtype="bfloat16",
        batch_size=BATCH_SIZE,
        sequence_lengths=SEQUENCE_LENGTHS,
        generate_new_tokens=GENERATE_NEW_TOKENS,
        num_runs=NUM_RUNS,
        gpu_environment=gpu_env,
    )

    for implementation in ATTENTION_IMPLEMENTATIONS:
        results.implementations.append(
            benchmark_implementation(implementation, tokenizer)
        )

    print_summary(results)
    output_path = save_results(results)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
