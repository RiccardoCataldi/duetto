"""
AMT's GPT-2 for one note at a time, fast: its KV cache is a fixed buffer and the step that feeds one token is a CUDA
graph, replayed with one launch. In eager mode the step is ~200 kernels the CPU launches one by one, and that
launching, not the GPU, is most of the time a note takes.

It is called as the Hugging Face model is (model(inp, past_key_values=past, use_cache=True)), for one duo at a time:
the cache it returns is a length into its buffer, and only the latest one is valid.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

STEPS = 8   # a few tokens are fed one replay at a time; more go in one eager pass


class Cache:
    """How many tokens of the buffer are the context; crop() forgets the ones after `length`."""

    def __init__(self, length=0):
        self.length = length

    def crop(self, length):
        self.length = min(self.length, length)


class GraphedGPT2:
    def __init__(self, model):
        config, gpt = model.config, model.transformer
        self.heads, self.size = config.n_head, config.n_positions
        self.dim = config.n_embd // config.n_head
        self.eps = config.layer_norm_epsilon
        self.wte, self.wpe = gpt.wte.weight, gpt.wpe.weight
        self.ln_f = (gpt.ln_f.weight, gpt.ln_f.bias)
        self.blocks = [(b.ln_1.weight, b.ln_1.bias, b.attn.c_attn.weight, b.attn.c_attn.bias,
                        b.attn.c_proj.weight, b.attn.c_proj.bias, b.ln_2.weight, b.ln_2.bias,
                        b.mlp.c_fc.weight, b.mlp.c_fc.bias, b.mlp.c_proj.weight, b.mlp.c_proj.bias)
                       for b in gpt.h]
        # what each layer's attention is scaled by: AMT also divides it by the layer's number
        self.scales = [self.dim ** -0.5 / (i + 1 if config.scale_attn_by_inverse_layer_idx else 1)
                       for i in range(len(self.blocks))]
        shape = (len(self.blocks), 1, self.heads, self.size, self.dim)
        self.keys = torch.zeros(shape, dtype=self.wte.dtype, device=self.wte.device)
        self.values = torch.zeros_like(self.keys)
        self.positions = torch.arange(self.size, device=self.wte.device)
        self.cache = Cache()
        self.token = torch.zeros(1, dtype=torch.long, device=self.wte.device)
        self.at = torch.zeros(1, dtype=torch.long, device=self.wte.device)
        self.graph = None
        if self.wte.is_cuda:
            self._capture()

    @torch.no_grad()
    def __call__(self, inp, past_key_values=None, use_cache=True):
        past = past_key_values.length if past_key_values is not None else 0
        tokens = inp[0]
        if len(tokens) > STEPS or self.graph is None:
            logits = self._forward(tokens, past)
        else:
            for i, token in enumerate(tokens):
                self.token.copy_(token.view(1))
                self.at.fill_(past + i)
                self.graph.replay()
            logits = self.logits.clone()
        self.cache = Cache(past + len(tokens))   # a session's older cache is now stale
        return SimpleNamespace(logits=logits.float().view(1, 1, -1), past_key_values=self.cache)

    def _forward(self, tokens, past):
        """Feeds `tokens` from position `past` in eager mode; the logits after the last one."""
        n = len(tokens)
        x = self.wte[tokens] + self.wpe[past:past + n]
        mask = self.positions[:past + n] <= self.positions[past:past + n, None]
        for layer, block in enumerate(self.blocks):
            x = self._block(x, layer, block, slice(past, past + n), past + n, mask)
        return self._head(x[-1])

    def _step(self):
        """Feeds self.token at position self.at: the step the graph records."""
        x = self.wte[self.token] + self.wpe[self.at]
        mask = (self.positions <= self.at)[None]
        for layer, block in enumerate(self.blocks):
            x = self._block(x, layer, block, self.at, self.size, mask)
        return self._head(x[0])

    def _block(self, x, layer, block, where, seen, mask):
        """One transformer layer over x (n, embd); writes its keys and values at `where` (a slice, or a position
        tensor in the graph) and attends to the first `seen` of them under `mask` (n, seen)."""
        ln1w, ln1b, attn_w, attn_b, proj_w, proj_b, ln2w, ln2b, fc_w, fc_b, out_w, out_b = block
        n = x.shape[0]
        q, k, v = torch.addmm(attn_b, F.layer_norm(x, x.shape[-1:], ln1w, ln1b, self.eps), attn_w).split(x.shape[-1], -1)
        q, k, v = (t.view(n, self.heads, self.dim).transpose(0, 1)[None] for t in (q, k, v))
        if isinstance(where, slice):
            self.keys[layer][:, :, where] = k
            self.values[layer][:, :, where] = v
        else:
            self.keys[layer].index_copy_(2, where, k)
            self.values[layer].index_copy_(2, where, v)
        a = F.scaled_dot_product_attention(q, self.keys[layer][:, :, :seen], self.values[layer][:, :, :seen],
                                           attn_mask=mask, scale=self.scales[layer])
        x = x + torch.addmm(proj_b, a[0].transpose(0, 1).reshape(n, -1), proj_w)
        h = F.gelu(torch.addmm(fc_b, F.layer_norm(x, x.shape[-1:], ln2w, ln2b, self.eps), fc_w), approximate='tanh')
        return x + torch.addmm(out_b, h, out_w)

    def _head(self, x):
        return F.layer_norm(x, x.shape[-1:], *self.ln_f, self.eps) @ self.wte.T

    def _capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(stream):
            for _ in range(3):   # warms up the kernels the graph will record
                self._step()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph):
            self.logits = self._step()
