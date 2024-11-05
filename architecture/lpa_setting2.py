import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states
    
class SiLUActivation(nn.Module):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.silu(input)
    
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
    
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )
    

class MyLlamaAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim, attn_lowdim, low_attn, max_position_embeddings=2048):
        super().__init__()
        self.hidden_size = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        
        if low_attn == 'yes':
            self.q_proj = nn.Sequential(
                nn.Linear(self.hidden_size, attn_lowdim, bias=False),
                nn.Linear(attn_lowdim, self.num_heads * self.head_dim, bias=False),)
            self.k_proj = nn.Sequential(
                nn.Linear(self.hidden_size, attn_lowdim, bias=False),
                nn.Linear(attn_lowdim, self.num_heads * self.head_dim, bias=False),)
            self.v_proj = nn.Sequential(
                nn.Linear(self.hidden_size, attn_lowdim, bias=False),
                nn.Linear(attn_lowdim, self.num_heads * self.head_dim, bias=False),)
            self.o_proj = nn.Sequential(
                nn.Linear(self.num_heads * self.head_dim, attn_lowdim, bias=False),
                nn.Linear(attn_lowdim, self.hidden_size, bias=False),)

        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, position_ids):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        # WARNING: padding mask is ignored, causal is always applied
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, dropout_p=0.0, is_causal=True,
        )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output


class MyLlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        ffn_lowdim: int,
        low_ffn: str,
    ):
        super().__init__()
        if low_ffn == 'yes':
            self.gate_proj = nn.Sequential(nn.Linear(hidden_size, ffn_lowdim, bias=False), nn.Linear(ffn_lowdim, intermediate_size, bias=False),)
            self.down_proj = nn.Sequential(nn.Linear(intermediate_size, ffn_lowdim, bias=False), nn.Linear(ffn_lowdim, hidden_size, bias=False),)
            self.up_proj = nn.Sequential(nn.Linear(hidden_size, ffn_lowdim, bias=False), nn.Linear(ffn_lowdim, intermediate_size, bias=False),)
        else:
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = SiLUActivation()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LPALayer(nn.Module):
    def __init__(self, embed_dim, ffn_dim, num_heads, head_dim, attn_lowdim, ffn_lowdim, low_attn, low_ffn):
        super().__init__()
        self.attention = MyLlamaAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            attn_lowdim=attn_lowdim,
            low_attn=low_attn,
            )
        self.norm1 = LlamaRMSNorm(embed_dim)
        self.norm2 = LlamaRMSNorm(embed_dim)
        self.ffn = MyLlamaMLP(
            hidden_size=embed_dim,
            intermediate_size=ffn_dim,
            ffn_lowdim=ffn_lowdim,
            low_ffn=low_ffn,
            )
        

    def forward(self, x, position_ids):
        # pre norm
        x = x + self.attention(self.norm1(x), position_ids)
        x = x + self.ffn(self.norm2(x))

        return x


class LPA(nn.Module):
    def __init__(self, vocab_size, embed_dim, ffn_dim, num_heads, head_dim, attn_lowdim, ffn_lowdim, num_layers, seq_len, low_attn, low_ffn):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            LPALayer(
                embed_dim=embed_dim, 
                ffn_dim=ffn_dim,
                num_heads=num_heads, 
                head_dim=head_dim,
                attn_lowdim=attn_lowdim,
                ffn_lowdim=ffn_lowdim,
                low_attn=low_attn,
                low_ffn=low_ffn,
                )
                for _ in range(num_layers)
        ])
        self.norm = LlamaRMSNorm(embed_dim)
        self.seq_len = seq_len

    def forward(self, x):
        device = x.device
        position_ids = torch.arange(0, self.seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, self.seq_len)

        x = self.token_embedding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, position_ids)

        x = self.norm(x)
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
            low_attn=low_attn,
            low_ffn=low_ffn,
            )

        self.fc = nn.Linear(embed_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids,
        labels,
        **kwargs
    ):  
        hidden_states = self.lpa(x=input_ids,)
        logits = self.fc(hidden_states)

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
    



class LPAForSequenceClassification(nn.Module):
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
        low_attn,
        low_ffn,
        num_labels,
        pad_token_id,
        config,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.num_labels = num_labels
        self.config = config

        self.LPA = LPA(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            attn_lowdim=attn_lowdim,
            ffn_lowdim=ffn_lowdim,
            num_layers=num_layers,
            seq_len=seq_len,
            low_attn=low_attn,
            low_ffn=low_ffn,
            )
        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(
        self,
        input_ids,
        labels,
        **kwargs
    ):
        hidden_states = self.LPA(x=input_ids,)
        logits = self.classifier(hidden_states)

        batch_size = input_ids.shape[0]
        sequence_lengths = (torch.eq(input_ids, self.pad_token_id).long().argmax(-1) - 1).to(logits.device)
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        

        if self.num_labels == 1:
            loss_fct = nn.MSELoss()
            loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))

        return [loss, pooled_logits]


from transformers import PretrainedConfig
class LPAConfig(PretrainedConfig):

    def __init__(
        self,
        hidden_size=1024,
    ):
        self.hidden_size = hidden_size
        super().__init__()