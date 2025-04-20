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


def compare_state_dicts(state_dict_1, state_dict_2, model_params):
    changed_params = []
    unchanged_params = []
    missing_in_state_dict_2 = []

    # Use model's parameter names as the reference
    param_names = list(model_params)

    total_params = len(param_names)
    compared_count = 0

    for name in param_names:
        if name not in state_dict_1:
            print(
                f"Parameter '{name}' not found in initial state_dict (this shouldn't happen)."
            )
            continue
        if name not in state_dict_2:
            # This case happens if the loaded state_dict is missing keys that are present in the base model.
            missing_in_state_dict_2.append(name)
            continue

        param1 = state_dict_1[name]
        param2 = state_dict_2[name]

        if not torch.equal(param1, param2):
            changed_params.append(name)
        else:
            unchanged_params.append(name)

        compared_count += 1

    print(f"Total parameters in model: {total_params}")
    print(f"Parameters compared: {compared_count}")
    print(f"Parameters CHANGED after loading: {len(changed_params)}")
    print(f"Parameters UNCHANGED after loading: {len(unchanged_params)}")

    if missing_in_state_dict_2:
        print(
            f"Parameters MISSING in loaded checkpoint (but present in base model): {len(missing_in_state_dict_2)}"
        )
        print("First 5 missing params:", missing_in_state_dict_2[:5])

    if changed_params:
        print("\nCHANGED parameters:")
        for name in changed_params:
            print(f"- {name} (Shape: {state_dict_1[name].shape})")
    else:
        print("\nNo parameters were changed by loading the state dict.")

    # Only show unchanged if some were changed
    if unchanged_params and changed_params:
        print("\nFirst 5 UNCHANGED parameters:")
        for name in unchanged_params[:5]:
            print(f"- {name} (Shape: {state_dict_1[name].shape})")

    return None


def main(args):
    model_wrapper = LayerTrainedLlama(args.base_model)
    initial_state_dict = {
        k: v.clone() for k, v in model_wrapper.model.state_dict().items()
    }

    weights_path = os.path.join(args.custom_model, "model.safetensors")
    state_dict = load_safetensors(weights_path)
    model_wrapper.load_state_dict(state_dict, strict=False)

    final_state_dict = {
        k: v.clone() for k, v in model_wrapper.model.state_dict().items()
    }

    assert initial_state_dict.keys() == final_state_dict.keys(), (
        "Mismatch in state_dict keys after loading."
    )

    compare_state_dicts(
        initial_state_dict, final_state_dict, model_wrapper.model.state_dict().keys()
    )
    print("Finished comparison.")


def parse_args():
    parser = argparse.ArgumentParser(description="Verify layer changes in Llama model.")

    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to the base model directory.",
    )

    parser.add_argument(
        "--custom_model",
        type=str,
        required=True,
        help="Path to the layer-trained model directory.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
