import argparse
import os
from functools import partial

import torch
import torch.nn as nn
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizerFast,
    Trainer,
    TrainingArguments,
)

from datasets import load_from_disk


class LlamaLayerTrainer(nn.Module):
    def __init__(self, model_path: str, target_layer_idx: int) -> None:
        super().__init__()
        # Load the full model
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
        # Llama tokenizer does not set a pad token by default
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.target_layer_idx = target_layer_idx
        target_layer = self.model.model.layers[target_layer_idx]

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze target layer
        for param in target_layer.parameters():
            param.requires_grad = True

        def initialize_weights(module):
            if isinstance(module, nn.Linear):
                # Kaiming normal initialization for weights
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

        # Recursively change weights
        target_layer.apply(initialize_weights)

        self.hook_output = None
        self._register_target_layer_hook()

    def _forward_hook(self, module, input, output) -> None:
        # We only need the hidden state, which is the first element.
        self.hook_output = output[0]

    def _register_target_layer_hook(self) -> None:
        target_module = self.model.model.layers[self.target_layer_idx]
        target_module.register_forward_hook(self._forward_hook)

    def forward(self, input_ids, attention_mask, **kwargs):
        self.hook_output = None
        # We return the full output object for simplicity
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        return outputs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )


def collate_fn(
    examples: list[dict], tokenizer: LlamaTokenizerFast
) -> dict[str, torch.Tensor]:
    texts = [ex["text_input"] for ex in examples]

    # Note that we must tokenize in the same manner as the dataset generation
    tokenized_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )

    target_hidden_states_list = []
    # Max sequence length in the batch
    max_len = tokenized_inputs["input_ids"].shape[1]
    for ex in examples:
        # Convert hidden states to tensor
        hs = torch.tensor(ex["decoder_output"], dtype=torch.float32)

        # Data generation was not batched, but our training will be done in batches.
        # This means that some of the decoder outputs can be shorter than required since our tokenizer is not working with a batched input.
        # I.e., we must pad (we include truncation here for generalization) the hidden states to the max length of the batch
        current_len = hs.shape[0]
        if current_len > max_len:
            # This case should not happen if max_length is consistent with how we generated the dataset
            raise ValueError("Dataset generation not consistent with assumptions")
        elif current_len < max_len:
            padding_size = max_len - current_len
            # Ensure padding is on the same device as hidden states
            padding = torch.zeros((padding_size, hs.shape[1]), dtype=hs.dtype)
            hs = torch.cat([hs, padding], dim=0)

        target_hidden_states_list.append(hs)

    # Stack hidden states, shape: (batch_size, seq_len, hidden_dim=4096)
    target_hidden_states = torch.stack(target_hidden_states_list)

    # Note we need the attention mask for the loss calculation
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "target_hidden_states": target_hidden_states,
    }


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        # Ensure target_hidden_states is on the correct device
        target_hidden_states = inputs["target_hidden_states"].to(
            self.accelerator.device
        )

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        # Extract the hidden states from the target layer
        actual_hidden_states = model.hook_output

        target_hidden_states = target_hidden_states.to(actual_hidden_states.dtype)

        loss_fct = nn.MSELoss(reduction="none")
        loss = loss_fct(actual_hidden_states, target_hidden_states)

        # Loss shape: (batch_size, seq_len, hidden_dim)
        # Attention mask shape: (batch_size, seq_len)
        mask = attention_mask.unsqueeze(-1).expand_as(loss)
        typed_mask = mask.to(loss.dtype)

        masked_loss = loss * typed_mask
        # We average over the non masked tokens (not padding)
        mean_loss = masked_loss.sum().float() / typed_mask.sum().float()

        return (mean_loss, outputs) if return_outputs else mean_loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        # Ensure model is in eval mode
        model.eval()

        # Separate target_hidden_states from inputs passed to the model
        inputs = self._prepare_inputs(inputs)
        target_hidden_states = inputs.pop("target_hidden_states")
        target_hidden_states = target_hidden_states.to(self.accelerator.device)

        with torch.no_grad():
            loss = self.compute_loss(
                model,
                inputs={**inputs, "target_hidden_states": target_hidden_states},
                return_outputs=False,
            )

        # The Trainer needs the 'loss' part for eval_loss metric.
        return (loss.detach(), None, None)


def main(args):
    # Create a Llama wrapper that zeroes out target layer, among other things
    model = LlamaLayerTrainer(args.model, args.index)
    dataset = load_from_disk(args.dataset)

    split = dataset.train_test_split(test_size=0.1, seed=137)
    train_dataset = split["train"]
    val_dataset = split["test"]

    training_args = TrainingArguments(
        output_dir=args.output,
        bf16=True,  # Use bf16 for mixed precision training
        logging_dir=args.logs,
        logging_steps=100,  # CHANGE
        report_to="tensorboard",
        remove_unused_columns=False,  # Ensure columns are kept in dataset as we use a custom data processor (collate_fn)
        # --- TRAINING ARGS ---
        max_steps=1000,
        save_strategy="steps",
        save_steps=100,  # CHANGE
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,  # Effective train batch = 8*4=32
        gradient_checkpointing=True,  # Reduce GPU memory usage which we need for this script
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # Required to work with our forward hooks
        # --- EVAL ARGS ---
        eval_strategy="steps",
        eval_steps=200,
        per_device_eval_batch_size=4,
        load_best_model_at_end=False,
        prediction_loss_only=False,
    )

    # Create partial function for collate_fn with the Llama tokenizer
    collate_fn_with_tokenizer = partial(collate_fn, tokenizer=model.tokenizer)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn_with_tokenizer,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=False)
    print("Training finished.")

    print("Saving the best model...")
    best_model_dir = os.path.join(args.output, f"llama_layer_{args.index}_best")
    trainer.save_model(best_model_dir)
    print("Model saved.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Llama layer.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model directory.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Index of the (decoder) layer to train.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        required=True,
        help="Path to save the tensorboard logs.",
    )


if __name__ == "__main__":
    main(parse_args())
