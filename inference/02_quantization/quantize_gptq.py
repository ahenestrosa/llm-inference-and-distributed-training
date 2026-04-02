"""
GPTQ quantization script: Quantize Llama 3.1 8B with different configs.

Produces a 4-bit quantized model with group_size=128 (standard).
Takes ~30-60 minutes on an A100 (Hessian computation).

Uses transformers' built-in GPTQConfig + optimum backend instead of
auto-gptq directly (which has build issues with modern Python).
"""

import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

MODEL_ID = "meta-llama/Llama-3.1-8B"
OUTPUT_DIR = Path(__file__).parent / "models" / "gptq"
NUM_CALIBRATION_SAMPLES = 128

CONFIGS: dict[str, dict] = {
    "4bit_g128": {
        "bits": 4,
        "group_size": 128,
        "damp_percent": 0.1,
        "desc_act": False,
    },
}


def prepare_calibration_data(
    num_samples: int = NUM_CALIBRATION_SAMPLES,
) -> list[str]:
    """Load calibration text samples from wikitext-2.

    GPTQConfig.dataset accepts a list of strings — tokenization is handled
    internally by the quantizer.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # Filter out empty lines and short texts
    texts = [t for t in dataset["text"] if len(t.strip()) > 50][:num_samples]

    print(f"Prepared {len(texts)} calibration samples")
    return texts


def quantize_model(config_name: str, config_params: dict) -> None:
    """Load FP16 model, quantize with GPTQ, and save."""
    output_path = OUTPUT_DIR / config_name

    if output_path.exists():
        print(f"  Skipping {config_name} — already exists at {output_path}")
        return

    print(f"\n{'='*50}")
    print(f"Quantizing: {config_name}")
    print(f"  bits={config_params['bits']}, group_size={config_params['group_size']}, "
          f"desc_act={config_params['desc_act']}")
    print(f"{'='*50}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Prepare calibration data
    print("  Preparing calibration data...")
    calibration_data = prepare_calibration_data()

    # Build GPTQConfig with calibration dataset
    gptq_config = GPTQConfig(
        bits=config_params["bits"],
        group_size=config_params["group_size"],
        damp_percent=config_params["damp_percent"],
        desc_act=config_params["desc_act"],
        dataset=calibration_data,
        tokenizer=tokenizer,
    )

    # Load and quantize in one step
    print("  Loading and quantizing (this will take 30-60 minutes)...")
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=gptq_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    elapsed = time.perf_counter() - start
    print(f"  Quantization completed in {elapsed / 60:.1f} minutes")

    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"  Saved to {output_path}")

    # Free memory
    del model
    torch.cuda.empty_cache()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for config_name, config_params in CONFIGS.items():
        quantize_model(config_name, config_params)

    print("\n" + "=" * 50)
    print("All quantizations complete!")
    print("=" * 50)
    for config_name in CONFIGS:
        path = OUTPUT_DIR / config_name
        print(f"  {config_name}: {path}")


if __name__ == "__main__":
    main()
