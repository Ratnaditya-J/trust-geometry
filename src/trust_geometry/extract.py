"""
GPU-side activation extraction for gpt-oss-20b (runs on the RunPod pod).

Produces, for each role-wrapped sequence:
  - mean-pooled hidden state over CONTENT tokens at every layer (0..num_layers),
  - the model's next-token logit features at the last content token (for the
    anti-circularity logit-baseline gate).

Also generates real (user -> analysis CoT -> final) conversations in the model's own
harmony format for the zero-shot-generalisation gate.

No statistics here — only tensors out. All analysis lives in analysis.py.
"""
from __future__ import annotations
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "openai/gpt-oss-20b"


def load(model_name=MODEL, token=None):
    tok = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda", token=token)
    model.eval()
    return tok, model


@torch.no_grad()
def activations_for(model, seqs, content_spans, logit_topk=64, batch_size=16, device="cuda"):
    """seqs: list[list[int]] token ids. content_spans: list[(start,end)].
    Returns:
      means: (N, L+1, d) float32 mean-pooled content hidden state per layer
      logit_feats: (N, logit_topk) top-k next-token logits at last content token
    """
    N = len(seqs)
    means, logit_feats = [], []
    for i in range(0, N, batch_size):
        chunk = seqs[i:i + batch_size]
        spans = content_spans[i:i + batch_size]
        maxlen = max(len(s) for s in chunk)
        ids = torch.full((len(chunk), maxlen), 0, dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for j, s in enumerate(chunk):
            ids[j, :len(s)] = torch.tensor(s); attn[j, :len(s)] = 1
        ids, attn = ids.to(device), attn.to(device)
        out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
        hs = out.hidden_states  # tuple (L+1) of (B, T, d)
        L = len(hs)
        for j, (st, en) in enumerate(spans):
            per_layer = np.stack([hs[l][j, st:en, :].float().mean(0).cpu().numpy() for l in range(L)])
            means.append(per_layer)
            last = en - 1
            lg = out.logits[j, last, :]
            topk = torch.topk(lg, logit_topk).values.float().cpu().numpy()
            logit_feats.append(topk)
        del out, hs
        torch.cuda.empty_cache()
    return np.stack(means).astype(np.float32), np.stack(logit_feats).astype(np.float32)


@torch.no_grad()
def generate_real_conversations(tok, model, user_prompts, max_new_tokens=200, device="cuda"):
    """Run the model on real user prompts; return, per conversation, the token ids and
    the content spans for the natural user / cot(analysis) / assistant(final) roles.
    Used for the zero-shot-generalisation gate (probes trained on neutral-tag data
    must recover role structure in the model's own natural text).
    """
    convos = []
    for up in user_prompts:
        msgs = [{"role": "user", "content": up}]
        prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(device)
        gen = model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or 199999)
        full = gen[0].tolist()
        text = tok.decode(full)
        convos.append({"user_prompt": up, "ids": full, "text": text})
    return convos
