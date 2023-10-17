---
date: 2023-09-29
readtime: 60
authors:
  - ludwig
categories:
  - 技术
---

# Llama源码阅读

[Llama](https://ai.meta.com/blog/large-language-model-llama-meta-ai/)[^llama]是由Meta设计，训练并开源的大语言模型。相比于GPT-3，Llama模型更小，但是训练更加充分，性能更强，是开源社区最受欢迎的大模型之一。

本文主要阅读[Huggingface的Llama实现](https://huggingface.co/docs/transformers/v4.31.0/model_doc/llama)，重点关注相对于最早的Transformer[^attention]，Llama采用了哪些新的技术和优化。

<!-- more -->

[跳转这里](#_3)直接开始源码阅读。

## 预备知识

### Transformer

你需要知道什么是Transformer，知道它是一种自注意力神经网络。

<figure markdown>
  ![原始的Transformer网络结构](./original_transformer.png){width="400"}

  原始的Transformer网络结构[^attention]
</figure>

强烈推荐阅读[原论文](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)，重点关注3.2节和3.3节对网络结构的描述。不过，如果在阅读了原论文之后你还是不确定它的实现也没关系，通过阅读Llama的代码，你会知道一个基于Transformer网络结构的语言模型是如何实现的。

### Pytorch

你需要知道基本的pytorch知识，知道它可以用于搭建神经网络。你可以在网上找一些最基本的pytorch教程，只要你能看懂下面的代码就可以了。

```python title="simple_network.py" linenums="1"
import torch
from torch import nn

# 一个简单的神经网络
class NeuralNetwork(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=6, out_dim=2):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.linear_relu_stack(x)

# 前向传播
net = NeuralNetwork()
input_tensor = torch.randn(10, 3)
output_tensor = net(input_tensor)
```

## 代码准备

本文所阅读的代码以下面的版本为准：

```txt title="requirements.txt" linenums="1"
torch==2.0.1
transformers==4.31.0
```

transformers库采用[单模型文件策略](https://huggingface.co/blog/zh/transformers-design-philosophy)，我们只需要阅读[`modeling_llama.py`](https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/llama/modeling_llama.py)即可。

## 源码阅读

### 了解代码结构

在阅读具体的实现前，应当对代码的整体结构逻辑有所了解。这一部分不需要搞清楚每一个细节，但是需要了解模型的实现代码是如何组织的。

#### 阅读大纲

我们首先观察一下`modeling_llama.py`文件里有哪些类和函数，在vscode中打开左边的大纲。

<figure markdown>
  ![模型文件大纲](./outline.png){width="400"}

  模型文件大纲
</figure>

不难猜测，LlamaModel类就是我们要找的模型主干，而LlamaAttention、LlamaMLP等类则是模型中具体的网络模块。更进一步，如果你对Transformer架构比较熟悉的话，可能会猜测LlamaDecoderLayer是每一层的Transformer网络，其中包含了LlamaAttention和LlamaMLP模块。

#### LlamaForCausalLM类

观察LlamaForCausalLM类

```python title="modeling_llama.py" linenums="727" hl_lines="6 9"
class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
```

可以看到它包含一个LlamaModel对象和一个线性的lm_head，后者用于计算下一个token的概率分布。

#### LlamaModel类

观察LlamaModel类

```python title="modeling_llama.py" linenums="547" hl_lines="14-16"
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
```

可以看到它包含一个Embedding层、一系列的LlamaDecoderLayer、和一个LlamaRMSNorm模块。

#### LlamaDecoderLayer类

观察llamaDecoderLayer类

```python title="modeling_llama.py" linenums="371" hl_lines="5-8"
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

其包括了Transformer结构中最重要的两个模块，即self-attention和FFN，分别是一个LlamaAttention对象和LlamaMLP对象，此外还有两个LlamaRMSNorm对象。

### 阅读具体实现

下面我们来阅读模型的具体实现，并且将重点放在Llama模型相对于最早的Transformer采用了哪些新的技术和优化。

#### RMSNorm

LayerNorm[^layernorm]是一种稳定深度神经网络训练的技术，通过对激活值向量进行normalize操作来稳定深度神经网络的训练。Llama使用的是RMSNorm[^rmsnorm]，去掉了re-centering的计算，效率高于经典的LayerNorm。

```python title="modeling_llama.py" linenums="75" hl_lines="13-15"
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

#### SwiGLU

在Transformer的FFN实现中，SwiGLU被证明是性能较好一种实现[^swiglu]。

```python title="modeling_llama.py" linenums="191" hl_lines="26"
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pretraining_tp = config.pretraining_tp
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj
```

可以看到Llama的MLP层使用了GLU的结构，即带有一个门控机制。

实际中会调整中间层的大小，来使得参数量和计算量与原始的FFN实现相当（常见的中间层维度是hidden_size的$\frac{8}{3}$倍左右，和原始4倍大小的中间层的参数量和计算量相当）。

#### Rotary Embedding

由于Transformer的Attention计算是不带有位置信息的，所以我们需要通过某种方法让模型能够感知到位置信息，这个技术被称为位置编码。位置编码的实现方式有很多种，从最早的绝对位置编码[^attention]，到相对位置编码，再到Llama使用的旋转位置编码[^rope]。

提出旋转位置编码的文章[^rope]在本blog撰写时候，公式部分的记号还是稍微有些混乱，需要静下心来看明白。

```python title="modeling_llama.py" linenums="173"
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
```

#### LlamaAttention

LlamaAttention的主要参数为四个hidden_size * hidden_size的映射矩阵，用于实现多头注意力。

```python title="modeling_llama.py" linenums="233" hl_lines="20-23"
class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()
```

现在我们走一遍Attention的前向计算过程。首先先明确每个Attention Block的输入维度为batch_size * seq_len * hidden_size

```python title="modeling_llama.py" linenums="278" hl_lines="10"
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
```

对输入进行映射，输出维度为batch_size * seq_len * hidden_size

```python title="modeling_llama.py" linenums="305"
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
```

拆分出每一个头的输出，此时输出维度变为batch_size * num_heads * seq_len * head_dim

```python title="modeling_llama.py" linenums="309"
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
```

在应用了旋转位置编码后，计算attn_weights，其维度为batch_size * num_heads * seq_len * seq_len

```python title="modeling_llama.py" linenums="330"
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
```

计算attention的输出，其维度为batch_size * num_heads * seq_len * head_dim

```python title="modeling_llama.py" linenums="345"
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
```

将多头输出cat到一起，注意这里需要使用contiguous，输出维度为batch_size * seq_len * hidden_size

```python title="modeling_llama.py" linenums="355"
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
```

最后将输出通过o_proj的映射，得到维度为batch_size * seq_len * hidden_size

```python title="modeling_llama.py" linenums="363"
            attn_output = self.o_proj(attn_output)
```

#### Pre-LayerNorm

原始的Transformer使用post-layernorm，研究表明pre-layernorm会使得训练更加稳定[^prenorm]。

<figure markdown>
  ![pre-layernorm](./pre_ln.png){width="400"}

  左侧为post-layernorm，右侧为pre-layernorm[^prenorm]
</figure>

可以看到pre-layernorm是加在Attention和MLP之前的：

```python title="modeling_llama.py" linenums="363"
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
```

#### Language modeling loss

当输入了labels时，LlamaForCausalLM会在每一个token位置计算下一个token的输出损失，注意这里logits和labels是需要错开一位的。

```python title="modeling_llama.py" linenums="827" hl_lines="4 5"
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
```

## 本文未讨论的内容

大模型的并行训练与推理，作为未来的学习计划

## 延伸阅读

[Andrej Karpathy的Youtube频道](https://www.youtube.com/@AndrejKarpathy)中有手撕GPT代码的教程，强烈推荐观看。

[Meta开源的llama代码](https://github.com/facebookresearch/llama)




[^llama]: Touvron et al. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (arXiv 2023)
[^attention]: Vaswani et al. [Attention Is All You Need.](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (NIPS 2017)
[^layernorm]: Ba et al. [Layer Normalization](https://arxiv.org/abs/1607.06450) (arXiv 2023)
[^rmsnorm]: Zhang et al. [Root Mean Square Layer Normalization](https://papers.nips.cc/paper_files/paper/2019/file/1e8a19426224ca89e83cef47f1e7f53b-Paper.pdf) (NIPS 2019)
[^swiglu]: Shazeer et al. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (arXiv 2020)
[^prenorm]: Xiong et al. [On Layer Normalization in the Transformer Architecture](http://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf) (ICML 2020)
[^rope]: Su et al. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (arXiv 2021)
