import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim, attn_lowdim, low_attn):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        if low_attn == 'yes':
            self.query_proj = nn.Sequential(nn.Linear(embed_dim, attn_lowdim), nn.Linear(attn_lowdim, num_heads * head_dim, bias=False),)
            self.key_proj = nn.Sequential(nn.Linear(embed_dim, attn_lowdim), nn.Linear(attn_lowdim, num_heads * head_dim, bias=False),)
            self.value_proj = nn.Sequential(nn.Linear(embed_dim, attn_lowdim), nn.Linear(attn_lowdim, num_heads * head_dim, bias=False),)
            self.out_proj = nn.Sequential(nn.Linear(num_heads * head_dim, attn_lowdim, bias=False), nn.Linear(attn_lowdim, embed_dim),)

        else:
            self.query_proj = nn.Linear(embed_dim, num_heads * head_dim)
            self.key_proj = nn.Linear(embed_dim, num_heads * head_dim)
            self.value_proj = nn.Linear(embed_dim, num_heads * head_dim)
            self.out_proj = nn.Linear(num_heads * head_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        q = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.out_proj(attn_output)


class LPALayer(nn.Module):
    def __init__(self, embed_dim, ffn_dim, num_heads, head_dim, attn_lowdim, ffn_lowdim, dropout, low_attn, low_ffn):
        super().__init__()
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            attn_lowdim=attn_lowdim,
            low_attn=low_attn,
            )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        if low_ffn == 'yes':
            self.fc = nn.Sequential(
                nn.Linear(embed_dim, ffn_lowdim),
                nn.Linear(ffn_lowdim, ffn_dim, bias=False),
                nn.ReLU(),
                nn.Linear(ffn_dim, ffn_lowdim, bias=False),
                nn.Linear(ffn_lowdim, embed_dim),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(),
                nn.Linear(ffn_dim, embed_dim),
            )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attention(x)
        x = x + self.dropout1(self.norm1(attn_output))
        fc_output = self.fc(x)
        x = x + self.dropout2(self.norm2(fc_output))
        return x


class LPA(nn.Module):
    def __init__(self, vocab_size, embed_dim, ffn_dim, num_heads, head_dim, attn_lowdim, ffn_lowdim, num_layers, seq_len, dropout, low_attn, low_ffn):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(seq_len, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            LPALayer(
                embed_dim=embed_dim, 
                ffn_dim=ffn_dim,
                num_heads=num_heads, 
                head_dim=head_dim,
                attn_lowdim=attn_lowdim,
                ffn_lowdim=ffn_lowdim,
                dropout=dropout,
                low_attn=low_attn,
                low_ffn=low_ffn,
                )
                for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.token_embedding(x) + self.positional_embedding(torch.arange(x.shape[1]).to(x.device))
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.fc(x)
        return x


class LPAForCausalLM(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        embed_dim,
        ffn_dim,
        num_heads, 
        head_dim,
        attn_lowdim, 
        ffn_lowdim,
        num_layers, 
        seq_len, 
        dropout,
        low_attn,
        low_ffn,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lpa = LPA(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            attn_lowdim=attn_lowdim,
            ffn_lowdim=ffn_lowdim,
            num_layers=num_layers,
            seq_len=seq_len,
            dropout=dropout,
            low_attn=low_attn,
            low_ffn=low_ffn,
            )

        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids,
        labels,
        **kwargs
    ):
        logits = self.lpa(x=input_ids,)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return [loss]