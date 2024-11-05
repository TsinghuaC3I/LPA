<div align="center">

<h1>Scalable Efficient Training of Large Language Models with Low-dimensional Projected Attention</h1>

</div>

🎉  This is the implementation of EMNLP 2024 main conference paper：[Scalable Efficient Training of Large Language Models with Low-dimensional Projected Attention](https://arxiv.org/abs/2411.02063)


## Preparation

### Prepare Data

In the paper/code, we use the WikiText-103 datasets and the Pile datasets, which are all open access on the Internet.

### Modify Path

Before running the code, please replace the following data or work path definition with your path:

- `LPA/scripts/lpa_train_setting1.sh` /  `LPA/scripts/lpa_train_setting2.sh`: `train_data_path`, `valid_data_path`, `test_data_path`, `tokenizer_path`, `output_dir`
- `LPA/model_train/lpa_train_setting1.py` / `LPA/model_train/lpa_train_setting2.py`: add your work path in Line 15
- `LPA/model_train/lpa_train_setting2.py`: add your wandb project name in Line 107


## LPA

You can apply LPA by running the following codes:

```bash
cd scripts
# model setting 1
bash lpa_train_setting1.sh
# model setting 2
bash lpa_train_setting2.sh
```

We explain some of the arguments as follows:

- `attn_lowdim`: The hyperparameters $r$ for the low-dimensional module applied in the attention layer.
- `ffn_lowdim`: The hyperparameters $r$ for the low-dimensional module applied in the FFN layer.
- `low_attn`: The argument to decide whether or not to apply low-dimensional module in the attention layer. Possible value is `yes` or `no`.
- `low_ffn`: The argument to decide whether or not to apply low-dimensional module in the FFN layer. Possible value is `yes` or `no`.

> The argument `low_attn` / `low_ffn` can decide whether or not to apply low-dimensional module in the attention / FFN layer. 
For LPA model, we set `low_attn` to 'yes' and `low_ffn` to 'no'. For original Transformer (our main baseline in the paper), we set `low_attn` and `low_ffn` to 'yes'.

## statement
Part of the code in file `LPA/architecture/lpa_setting2.py` refers to the implementation of LLaMA model in Huggingface Transformer.

## Bugs or questions?

If you have any questions related to the codes or the paper, please contact Xingtai Lv (`lvxt24@mails.tsinghua.edu.cn`) or open an issue.

## Citation

If you find our work useful, please use the following citation: 

```bibtex
@misc{lv2024scalableefficienttraininglarge,
      title={Scalable Efficient Training of Large Language Models with Low-dimensional Projected Attention}, 
      author={Xingtai Lv and Ning Ding and Kaiyan Zhang and Ermo Hua and Ganqu Cui and Bowen Zhou},
      year={2024},
      eprint={2411.02063},
      archivePrefix={arXiv},
      primaryClass={cs.CL}, 
}
```