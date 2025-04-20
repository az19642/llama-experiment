import argparse
import os

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors
from transformers import LlamaForCausalLM, LlamaTokenizerFast


class LayerTrainedLlama(nn.Module):
    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        self.tokenizer = LlamaTokenizerFast.from_pretrained(model_path)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        return outputs


def _load_model(
    base_model_path: str,
    saved_model_path: str,
) -> LayerTrainedLlama:
    model_wrapper = LayerTrainedLlama(base_model_path)
    if saved_model_path is not None:
        weights_path = os.path.join(saved_model_path, "model.safetensors")
        state_dict = load_safetensors(weights_path)
        model_wrapper.load_state_dict(state_dict, strict=False)

    model_wrapper.eval()
    return model_wrapper


def main(args) -> None:
    loaded_model = _load_model(args.base_model, args.custom_model)
    tokenizer = loaded_model.tokenizer

    inputs = tokenizer(args.prompt, return_tensors="pt").to(loaded_model.model.device)

    # Perform inference
    print("Generating...")
    with torch.no_grad():
        output = loaded_model.model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
        )

    print(f"Generated output: {output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repeated inference (generation) with the Llama-2-7b-hf model"
    )

    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to the base model directory",
    )

    parser.add_argument(
        "--custom-model",
        type=str,
        default=None,
        help="Path to the custom model directory. If specified, it will be used instead of the base model.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        required=True,
        help="Maximum number of new tokens to generate",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
