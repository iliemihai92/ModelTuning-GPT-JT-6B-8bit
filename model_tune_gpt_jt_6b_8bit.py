from sklearn.model_selection import train_test_split

import transformers

import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import custom_fwd, custom_bwd

from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise

from tqdm.auto import tqdm
import transformers

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import custom_fwd, custom_bwd

from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise

from tqdm.auto import tqdm

from transformers import (AutoTokenizer, AdamW, get_scheduler)
import torch
import pandas as pd
from datasets import Dataset
from argparse import ArgumentParser
from bitsandbytes.optim import Adam8bit

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from datasets import Dataset as HDataset
from tqdm import tqdm


class FrozenBNBLinear(nn.Module):
    def __init__(self, weight, absmax, code, bias=None):
        assert isinstance(bias, nn.Parameter) or bias is None
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None
        self.bias = bias
 
    def forward(self, input):
        output = DequantizeAndLinear.apply(input, self.weight, self.absmax, self.code, self.bias)
        if self.adapter:
            output += self.adapter(input)
        return output
 
    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FrozenBNBLinear":
        weights_int8, state = quantize_blockise_lowmemory(linear.weight)
        return cls(weights_int8, *state, linear.bias)
 
    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"
 
 
class DequantizeAndLinear(torch.autograd.Function): 
    @staticmethod
    @custom_fwd
    def forward(ctx, input: torch.Tensor, weights_quantized: torch.ByteTensor,
                absmax: torch.FloatTensor, code: torch.FloatTensor, bias: torch.FloatTensor):
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        ctx.save_for_backward(input, weights_quantized, absmax, code)
        ctx._has_bias = bias is not None
        return F.linear(input, weights_deq, bias).clone()
 
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
 
 
class FrozenBNBEmbedding(nn.Module):
    def __init__(self, weight, absmax, code):
        super().__init__()
        self.num_embeddings, self.embedding_dim = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None
 
    def forward(self, input, **kwargs):
        with torch.no_grad():
            # note: both quantuized weights and input indices are *not* differentiable
            weight_deq = dequantize_blockwise(self.weight, absmax=self.absmax, code=self.code)
            output = F.embedding(input, weight_deq, **kwargs)
        if self.adapter:
            output += self.adapter(input)
        return output 
 
    @classmethod
    def from_embedding(cls, embedding: nn.Embedding) -> "FrozenBNBEmbedding":
        weights_int8, state = quantize_blockise_lowmemory(embedding.weight)
        return cls(weights_int8, *state)
 
    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_embeddings}, {self.embedding_dim})"
 
 
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
 
 
def convert_to_int8(model):
    """Convert linear and embedding modules to 8-bit with optional adapters"""
    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                print(name, child)
                setattr(
                    module,
                    name,
                    FrozenBNBLinear(
                        weight=torch.zeros(child.out_features, child.in_features, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                        bias=child.bias,
                    ),
                )
            elif isinstance(child, nn.Embedding):
                setattr(
                    module,
                    name,
                    FrozenBNBEmbedding(
                        weight=torch.zeros(child.num_embeddings, child.embedding_dim, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                    )
                )

class GPTJBlock(transformers.models.gptj.modeling_gptj.GPTJBlock):
    def __init__(self, config):
        super().__init__(config)

        convert_to_int8(self.attn)
        convert_to_int8(self.mlp)


class GPTJModel(transformers.models.gptj.modeling_gptj.GPTJModel):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)
        

class GPTJForCausalLM(transformers.models.gptj.modeling_gptj.GPTJForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)


transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock

class T5ForConditionalGeneration(transformers.models.t5.modeling_t5.T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)

transformers.models.t5.modeling_t5.T5ForConditionalGeneration = T5ForConditionalGeneration

config = transformers.GPTJConfig.from_pretrained("iliemihai/GPT-JT-6B-v1-8bit")
tokenizer = transformers.AutoTokenizer.from_pretrained("iliemihai/GPT-JT-6B-v1-8bit")

config.pad_token_id = config.eos_token_id
tokenizer.pad_token = config.pad_token_id

gpt = GPTJForCausalLM.from_pretrained("iliemihai/GPT-JT-6B-v1-8bit", low_cpu_mem_usage=True)

from google.colab import drive
drive.mount('/content/drive')
%cd drive/MyDrive/GPT_JT_8bit

from sklearn.model_selection import train_test_split

df = pd.read_csv("./train.csv")
df = df.drop('UsedByPublishedData', axis=1)
df.dropna(inplace=True)
df["label"] = df.NewsType
print("LABELS:", set(list(df["label"])))
new_df = df[['Body',  'label']].copy()
new_df["Body"] = df[["Headline", "Body"]].apply(" ".join, axis=1)
new_df["text"] = new_df["Body"]
new_df = new_df.drop('Body', axis=1)
print("LABELS:", set(list(new_df["label"])), len(new_df["label"]))
new_df.dropna(inplace=True)
print("LABELS:", set(list(new_df["label"])), len(new_df["label"]))

train, test = train_test_split(new_df, test_size=0.2, random_state=42, shuffle=True)from google.colab import drive
#drive.mount('/content/drive')
#%cd drive/MyDrive/GPT_JT_8bit

from sklearn.model_selection import train_test_split

df = pd.read_csv("./train.csv")
df = df.drop('UsedByPublishedData', axis=1)
df.dropna(inplace=True)
df["label"] = df.NewsType
print("LABELS:", set(list(df["label"])))
new_df = df[['Body',  'label']].copy()
new_df["Body"] = df[["Headline", "Body"]].apply(" ".join, axis=1)
new_df["text"] = new_df["Body"]
new_df = new_df.drop('Body', axis=1)
print("LABELS:", set(list(new_df["label"])), len(new_df["label"]))
new_df.dropna(inplace=True)
print("LABELS:", set(list(new_df["label"])), len(new_df["label"]))

train, test = train_test_split(new_df, test_size=0.2, random_state=42, shuffle=True)

class NewsDataset(Dataset):
    def __init__(self, df):
        self.instances = []

        dataset = df
        for line, label in tqdm(zip(dataset["text"].values, dataset["label"].values), total=len(dataset["label"].values)):
            instance = {
                        "sentence": line+"\nLABEL: "+label+"<|endoftext|>",
                       }
            self.instances.append(instance)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        return self.instances[i]

def my_collate(batch):
    sentences = []
    for instance in batch:
        sentences.append(instance["sentence"])

    sentences_batch = tokenizer(sentences, padding="max_length", max_length=2048, truncation=True, return_tensors="pt")
    sentences_batch = {key:val.to("cuda") for key, val in zip(sentences_batch.keys(), sentences_batch.values())}
    return sentences_batch

train_dataset = NewsDataset(train)
test_dataset = NewsDataset(test)

train_dataloader = DataLoader(train_dataset, batch_size=2,  shuffle=True, collate_fn=my_collate, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, num_workers=4, shuffle=False, collate_fn=my_collate, pin_memory=True, drop_last=True )

def add_adapters(model, adapter_dim=4, p = 0.1):
    assert adapter_dim > 0

    for name, module in model.named_modules():
      if isinstance(module, FrozenBNBLinear):
          if "attn" in name or "mlp" in name or "head" in name:
              print("Adding adapter to", name)
              module.adapter = nn.Sequential(
                nn.Linear(module.in_features, adapter_dim, bias=False),
                nn.Dropout(p=p),
                nn.Linear(adapter_dim, module.out_features, bias=False),
            )
              print("Initializing", name)
              nn.init.zeros_(module.adapter[2].weight)

          else:
              print("Not adding adapter to", name)
      elif isinstance(module, FrozenBNBEmbedding):
          print("Adding adapter to", name)
          module.adapter = nn.Sequential(
                nn.Embedding(module.num_embeddings, adapter_dim),
                nn.Dropout(p=p),
                nn.Linear(adapter_dim, module.embedding_dim, bias=False),
            )
          print("Initializing", name)
          nn.init.zeros_(module.adapter[2].weight)

add_adapters(gpt)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpt.to(device)

gpt.config.max_length=2048
gpt.config.max_length

from bitsandbytes.optim import Adam8bit

gpt.gradient_checkpointing_enable()
optimizer = Adam8bit(gpt.parameters(), lr=1e-5, weight_decay=0.01)

num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer, int(num_training_steps*0.1), num_training_steps
)

filepath = '/content/drive/MyDrive/GPT_JT_8bit/model.pt'

from tqdm.auto import tqdm

scaler = torch.cuda.amp.GradScaler()
progress_bar = tqdm(range(num_training_steps))
gpt.train()
gpt.gradient_checkpointing_enable()
k = 0

for epoch in range(num_epochs):
    for batch in train_dataloader:

        k = k + 1
        if k % 500 == 0:
          print(k)
          state = {'k' : k, 'epoch': num_epochs, 'lr_scheduler': lr_scheduler.state_dict(), 'state_dict': gpt.state_dict(), 'optimizer': optimizer.state_dict()}
          torch.save(state, filepath)

        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
          out = gpt.forward(**batch,use_cache=False)
          loss = F.cross_entropy(out.logits[:, :-1, :].flatten(0, -2), batch['input_ids'][:, 1:].flatten(),
                                reduction='mean', label_smoothing=0.1)
        
        if k % 100 == 0:
            print(loss)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step()
        progress_bar.update(1)


gpt.eval()
with torch.no_grad():
  prompt = tokenizer("Quote:", truncation=True, padding=True, max_length=128, return_tensors='pt')
  prompt = {key: value.to(device) for key, value in prompt.items()}
  out = gpt.generate(**prompt, max_length=128, top_k=50, top_p=0.9, temperature=1.0, do_sample=True, repetition_penalty = 1.2, num_beams=1)
  print(tokenizer.decode(out[0]))
