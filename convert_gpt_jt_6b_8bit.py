import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch.cuda.amp import custom_fwd, custom_bwd
from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
import transformers
#%config Completer.use_jedi = False


model_name = "togethercomputer/GPT-JT-6B-v1"
gpt = transformers.AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

def quantize_blockise_lowmemory(matrix: torch.Tensor, chunk_size: int = 2 ** 20):
    assert chunk_size % 4096 == 0
    code = None
    chunks = []
    absmaxes = []
    flat_tensor = matrix.view(-1)
    for i in range((matrix.numel() - 1) // chunk_size + 1):
        input_chunk = flat_tensor[i * chunk_size: (i + 1) * chunk_size].clone()
        quantized_chunk, (absmax_chunk, code) = quantize_blockwise(input_chunk, code=code)
        chunks.append(quantized_chunk)
        absmaxes.append(absmax_chunk)
        
    matrix_i8 = torch.cat(chunks).reshape_as(matrix)
    absmax = torch.cat(absmaxes)
    return matrix_i8, (absmax, code)

class DequantizeAndLinear(torch.autograd.Function):
    
    @staticmethod
    @custom_fwd
    def forward(ctx, input: torch.Tensor, weights_quantized: torch.ByteTensor,
                absmax: torch.FloatTensor, code: torch.FloatTensor, bias: torch.FloatTensor):
        
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        ctx.save_for_backward(input, weights_quantized, absmax, code)
        ctx._has_bias = bias is not None
        return F.linear(input, weights_deq, bias)
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        assert not ctx.needs_input_grad[1] and not ctx.needs_input_grad[2] and not ctx.needs_input_grad[3]
        input, weights_quantized, absmax, code = ctx.saved_tensors
        # grad_output: [*batch, out_features]
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        grad_input = grad_output @ weights_deq
        grad_bias = grad_output.flatten(0, -2).sum(dim=0) if ctx._has_bias else None
        return grad_input, None, None, None, grad_bias


class BNBLinearWithAdapter(nn.Module):
    def __init__(self, weight, absmax, code,  bias=None, adapter_dim=0):
        assert isinstance(bias, nn.Parameter) or bias is None
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.bias = bias
        
        if adapter_dim > 0:
            self.adapter = nn.Sequential(
                nn.Linear(self.in_features, adapter_dim, bias=False),
                nn.Linear(adapter_dim, self.out_features, bias=False),
            )
            
            nn.init.zeros_(self.adapter[1].weight)
        else:
            self.adapter = None
        
    def forward(self, input):
        out = DequantizeAndLinear.apply(input, self.weight, self.absmax, self.code, self.bias)
        
        if self.adapter:
            return self.adapter(input) + out
            
        return out
        
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "FrozenBNBLinear":
        weights_int8, state = quantize_blockise_lowmemory(linear.weight)
        return cls(weights_int8, *state, linear.bias, **kwargs)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"


class BNBEmbeddingWithAdapter(nn.Module):
    def __init__(self, weight, absmax, code, adapter_dim=0):
        super().__init__()
        self.num_embeddings, self.embedding_dim = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        
        if adapter_dim > 0:
            self.adapter = nn.Sequential(
                nn.Embedding(self.num_embeddings, adapter_dim),
                nn.Linear(adapter_dim, self.embedding_dim, bias=False),
            )
            
            nn.init.zeros_(self.adapter[1].weight)
        else:
            self.adapter = None
        
    def forward(self, input, **kwargs):
        with torch.no_grad():
            # note: both quantuized weights and input indices are *not* differentiable
            weight_deq = dequantize_blockwise(self.weight, absmax=self.absmax, code=self.code)
            out = F.embedding(input, weight_deq, **kwargs)
        if self.adapter:
            return out + self.adapter(input, **kwargs)
        
        return out
    
    @classmethod
    def from_embedding(cls, embedding: nn.Embedding, **kwargs) -> "FrozenBNBEmbedding":
        weights_int8, state = quantize_blockise_lowmemory(embedding.weight)
        return cls(weights_int8, *state, **kwargs)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_embeddings}, {self.embedding_dim})"

def bnbfy_(model, adapter_dim: int = 0):
    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                print(name, child)
                setattr(module, name, BNBLinearWithAdapter.from_linear(child, adapter_dim=adapter_dim))
                
            elif isinstance(child, nn.Embedding):
                print(name, child)
                setattr(module, name, BNBEmbeddingWithAdapter.from_embedding(child, adapter_dim=adapter_dim))

bnbfy_(gpt, adapter_dim=0)

prompt = tokenizer("A cat sat on a mat and", return_tensors='pt')
out = gpt.generate(**prompt, min_length=8, max_length=16, do_sample=True)
tokenizer.decode(out[0])

path = "model.bin"
torch.save(gpt.state_dict(), path)