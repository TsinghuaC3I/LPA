from transformers import (
    LlamaTokenizer, 
    TrainingArguments,
    Trainer,
    default_data_collator,
    HfArgumentParser,
)
from datasets import load_dataset
from itertools import chain
from dataclasses import dataclass, field
import math
import os

import sys
sys.path.append('your work path')
from architecture.lpa_setting1 import LPAForCausalLM

@dataclass
class DataArguments:
    train_data_file: str = field()
    valid_data_file: str = field()
    test_data_file: str = field()
    tokenizer_path: str = field()
    block_size: int = field()

@dataclass
class ModelArguments:
    embed_dim: int = field()
    ffn_dim: int = field()
    num_heads: int = field()
    head_dim: int = field()
    attn_lowdim: int = field()
    ffn_lowdim: int = field()
    num_layers: int = field()
    dropout: float = field()
    low_attn: str = field()
    low_ffn: str = field()

def main():
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    rank = int(os.environ["RANK"])
    if rank == 0:
        print(data_args)
        print(model_args)
        print(train_args)

    block_size = data_args.block_size

    tokenizer = LlamaTokenizer.from_pretrained(data_args.tokenizer_path)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left" 

    train_data = load_dataset("text", data_files=data_args.train_data_file)
    column_names = train_data["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    train_tokenized_datasets = train_data.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # The final part that is not enough for block_size is directly thrown away
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    train_dataset = train_tokenized_datasets.map(
            group_texts,
            batched=True,
            desc=f"Grouping texts in chunks of {block_size}",
        )['train']


    valid_data = load_dataset("text", data_files=data_args.valid_data_file)
    column_names = valid_data["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    valid_tokenized_datasets = valid_data.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )
    valid_dataset = valid_tokenized_datasets.map(
            group_texts,
            batched=True,
            desc=f"Grouping texts in chunks of {block_size}",
        )['train']

    low_model = LPAForCausalLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=model_args.embed_dim,
        ffn_dim=model_args.ffn_dim,
        num_heads=model_args.num_heads,
        head_dim=model_args.head_dim,
        attn_lowdim=model_args.attn_lowdim, 
        ffn_lowdim=model_args.ffn_lowdim,
        num_layers=model_args.num_layers,
        seq_len=block_size,
        dropout=model_args.dropout,
        low_attn=model_args.low_attn,
        low_ffn=model_args.low_ffn,
    )

    print(low_model)
    # compute the number of model parameters
    num_params = 0
    for param in low_model.parameters():
        num_params += param.numel()
    print(f"Number of parameters: {num_params}")
    
    # train
    trainer = Trainer(
        model=low_model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=train_args.output_dir)


    # test
    test_data = load_dataset("text", data_files=data_args.test_data_file)
    column_names = test_data["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    test_tokenized_datasets = test_data.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )
    test_dataset = test_tokenized_datasets.map(
            group_texts,
            batched=True,
            desc=f"Grouping texts in chunks of {block_size}",
        )['train']

    # evaluation: compute test ppl
    eval_results = trainer.evaluate(eval_dataset=test_dataset)
    eval_loss = eval_results["eval_loss"]
    perplexity = math.exp(eval_loss)
    if rank == 0:
        print(f"loss: {eval_loss}")
        print(f"perplexity: {perplexity}")

if __name__ == "__main__":
    main()