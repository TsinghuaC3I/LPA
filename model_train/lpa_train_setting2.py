from transformers import (
    LlamaTokenizer, 
    TrainingArguments,
    Trainer,
    default_data_collator,
    HfArgumentParser,
)
from datasets import load_dataset, load_from_disk
from dataclasses import dataclass, field
import wandb
import math
import os

import sys
sys.path.append('your work path')
from architecture.lpa_setting2 import LPAForCausalLM

@dataclass
class DataArguments:
    train_data_path: str = field()
    valid_data_path: str = field()
    test_data_path: str = field()
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
    low_attn: str = field()
    low_ffn: str = field()

def main():
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    rank = int(os.environ["RANK"])
    os.makedirs(train_args.output_dir, exist_ok=True)

    if rank == 0:
        print(data_args)
        print(model_args)
        print(train_args)

    block_size = data_args.block_size

    tokenizer = LlamaTokenizer.from_pretrained(data_args.tokenizer_path)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left" 

    from torch.utils.data import IterableDataset
    class MyValDataset(IterableDataset):
        def __init__(self, data_path):
            self.data_path = data_path
        def __iter__(self):
            for batch in load_from_disk(self.data_path):
                yield batch
    class MyTrainDataset(IterableDataset):
        def __init__(self, data_path):
            self.data_path = data_path
        def __iter__(self):
            for index in range(100):
                for batch in load_from_disk(self.data_path+'/'+str(index)):
                    yield batch
    
            
    train_dataset = MyTrainDataset(data_args.train_data_path)
    valid_dataset = MyValDataset(data_args.valid_data_path)

    low_model = LPAForCausalLM(
        vocab_size=tokenizer.vocab_size+1,
        embed_dim=model_args.embed_dim,
        ffn_dim=model_args.ffn_dim,
        num_heads=model_args.num_heads,
        head_dim=model_args.head_dim,
        attn_lowdim=model_args.attn_lowdim, 
        ffn_lowdim=model_args.ffn_lowdim,
        num_layers=model_args.num_layers,
        seq_len=block_size,
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

    if rank == 0 and train_args.report_to == ['wandb']:
        wandb.init(project="your wandb project name", name=train_args.run_name)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=train_args.output_dir)


    # test
    test_dataset = MyValDataset(data_args.test_data_path)
    # evaluation: compute test ppl
    eval_results = trainer.evaluate(eval_dataset=test_dataset)
    eval_loss = eval_results["eval_loss"]
    perplexity = math.exp(eval_loss)
    if rank == 0:
        print(f"loss: {eval_loss}")
        print(f"perplexity: {perplexity}")

if __name__ == "__main__":
    main()