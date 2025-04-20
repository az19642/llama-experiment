import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizerFast

from datasets import Dataset, load_dataset
from datasets.utils import disable_progress_bar


def _load_model(model_path: str) -> tuple[LlamaForCausalLM, LlamaTokenizerFast]:
    """Load the model from disk."""
    tokenizer = LlamaTokenizerFast.from_pretrained(model_path, local_files_only=True)
    # Pad token is not set by default
    tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    return model, tokenizer


def _load_dataset(dataset_path: str) -> Dataset:
    """Load the dataset from disk. Assumes the files are in jsonl.zst format."""
    dataset = load_dataset(
        "json", data_files=f"{dataset_path}/*.jsonl.zst", keep_in_memory=False
    )
    # dataset = load_from_disk(dataset_path, keep_in_memory=False)
    return dataset["train"]


def _attach_hooks(layers):
    """Attach forward hooks to the decoder layers of the model."""
    decoder_outputs = defaultdict(list)

    def hook_fn(layer_id):
        def fn(module, input, output):
            # output[0] is of shape (batch_size, seq_len, hidden_dim)
            output_sequence = output[0]
            # convert back to float16 so we can convert to numpy
            if output_sequence.dtype == torch.bfloat16:
                output_sequence = output_sequence.to(torch.float16)
            # Store the entire batch output
            decoder_outputs[layer_id].append(output_sequence.cpu().numpy())

        return fn

    hooks = []
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(hook_fn(i)))

    return hooks, decoder_outputs


def _save_as_dataset(
    output_dir: str,
    decoder_outputs: dict[int, list[torch.Tensor]],
    inputs: list[str],
    chunk_idx: int,
) -> None:
    """Map the raw text inputs to the decoder outputs and save as a dataset."""
    for layer_id, outputs in decoder_outputs.items():
        layer_dir = os.path.join(output_dir, f"layer_{layer_id}")
        os.makedirs(layer_dir, exist_ok=True)

        # Concatenate all outputs across the batch dimension
        # concatenated_outputs = torch.cat(outputs, dim=0)
        concatenated_outputs = np.concatenate(outputs, axis=0)

        dataset_dict = {
            "text_input": inputs,
            "decoder_output": concatenated_outputs,
        }
        dataset = Dataset.from_dict(dataset_dict)

        dataset_path = os.path.join(layer_dir, f"chunk_{chunk_idx}")
        dataset.save_to_disk(dataset_path)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output, exist_ok=True)

    model, tokenizer = _load_model(args.model)

    # Disable datasets progress bar
    disable_progress_bar()

    dataset = _load_dataset(args.dataset)
    hooks, decoder_outputs = _attach_hooks(model.model.layers)

    max_examples = len(dataset["text"])
    if args.max_examples > 0:
        max_examples = min(max_examples, args.max_examples)

    model.eval()
    with torch.inference_mode():
        inputs_processed = []
        chunk_idx = 0
        for batch_start in tqdm(
            range(0, max_examples, args.batch_size),
            desc=f"Processing {max_examples} examples in batches",
        ):
            batch_end = min(batch_start + args.batch_size, max_examples)
            inputs_batch = dataset["text"][batch_start:batch_end]
            inputs_processed.extend(inputs_batch)

            inputs_tokenized = tokenizer(
                inputs_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(device)
            del inputs_batch

            _ = model(**inputs_tokenized)
            del inputs_tokenized

            # Save at specified buffer_size/frequency, default is every batch
            if len(inputs_processed) >= args.buffer_size or batch_end >= max_examples:
                _save_as_dataset(
                    args.output, decoder_outputs, inputs_processed, chunk_idx
                )
                decoder_outputs.clear()
                inputs_processed.clear()
                chunk_idx += 1

    for hook in hooks:
        hook.remove()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create dataset of hidden states from the Llama-2-7b-hf model"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model directory",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset directory",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the dataset generated from the decoder outputs",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        required=True,
        help="Maximum number of examples to process (-1 for all)",
    )

    parser.add_argument(
        "--buffer-size",
        type=int,
        required=True,
        help="Number of examples to process in memory before writing to disk",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Batch size for batched inference",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        required=True,
        help="Maximum sequence tokenization length",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
