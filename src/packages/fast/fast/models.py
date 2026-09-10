"""Consistent small-model loading across labs.

Deliberately thin. Add helpers here when a second lab needs them — not before.
"""

from __future__ import annotations

__all__ = ["SMALL_MODELS", "chat", "load_default", "load_model"]

# Vetted for the program: ungated, small enough for a mid-tier GPU, instruction-tuned.
# Anything added here must be checked for licence and gating first — 20 people hitting
# a gated repo on the morning of Day 3 is a bad way to find out.
SMALL_MODELS = {
    "smol-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",  # gated: needs an accepted licence
    "gemma-2b": "google/gemma-2-2b-it",  # gated: needs an accepted licence
}


def load_model(name: str, dtype: str = "auto", device_map: str = "auto", **kwargs):
    """Load a model and tokenizer by short name (see SMALL_MODELS) or full HF id."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = SMALL_MODELS.get(name, name)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=device_map, **kwargs
    )
    model.eval()
    return model, tokenizer



def load_default():
    """Load the small instruction model the Day 1 labs share.

    Qwen 0.5B in the room; a 135M stand-in under CI, where there's no GPU and the numbers
    don't need to mean anything, only the code paths do. Runs on CPU in a couple of minutes.
    """
    import torch

    from fast.colab import ci_mode

    name = "smol-135m" if ci_mode() else "qwen-0.5b"
    dtype = "auto" if torch.cuda.is_available() else "float32"
    model, tokenizer = load_model(name, dtype=dtype)
    print(f"loaded {name} on {model.device}")
    return model, tokenizer

def chat(model, tokenizer, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
    """Single-turn chat completion. Returns only the newly generated text."""
    messages = [{"role": "user", "content": prompt}]
    # return_dict=True gives input_ids *and* attention_mask. Without the mask, generation
    # is subtly wrong on padded batches and noisy about it on single prompts.
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, **kwargs
    )
    prompt_length = inputs["input_ids"].shape[-1]
    return tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
