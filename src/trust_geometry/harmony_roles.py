"""
Tag-controlled role corpus for gpt-oss-20b (native harmony format).

Faithful reproduction of the role-probe training data from "Prompt Injection as
Role Confusion" (arXiv:2603.12277), Section 4.1 / G.1:

  - Identical neutral content is wrapped in each of the five role tags so the probe
    can only learn the *tag-induced* geometry, not semantics/style/position.
  - Content-token positions are held IDENTICAL across all role variants (the paper's
    positional control): we left-pad each variant with filler tokens so the content
    span always begins at the same absolute index.
  - We record the exact content-token span so activation extraction uses CONTENT
    tokens only, excluding role-tag / control tokens (paper G.2).

gpt-oss uses the "harmony" format. The five source-paper roles map to native
harmony constructs as:
    system    -> <|start|>system<|message|> ... <|end|>
    user      -> <|start|>user<|message|> ... <|end|>
    cot        -> <|start|>assistant<|channel|>analysis<|message|> ... <|end|>   (the CoT / <think>)
    assistant -> <|start|>assistant<|channel|>final<|message|> ... <|end|>
    tool      -> <|start|>functions.lookup to=assistant<|channel|>commentary<|message|> ... <|end|>

Ordered authority hierarchy under test (CLAIM 2): system > assistant > cot > user > tool
(we do NOT assume this ordering in the geometry; it is recovered/tested, and also
measured behaviourally to de-circularise — see prereg).
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import Optional

ROLES = ["system", "user", "cot", "assistant", "tool"]

# native harmony wrappers (prefix before content, suffix after content)
ROLE_WRAP = {
    "system":    ("<|start|>system<|message|>", "<|end|>"),
    "user":      ("<|start|>user<|message|>", "<|end|>"),
    "cot":       ("<|start|>assistant<|channel|>analysis<|message|>", "<|end|>"),
    "assistant": ("<|start|>assistant<|channel|>final<|message|>", "<|end|>"),
    "tool":      ("<|start|>functions.lookup to=assistant<|channel|>commentary<|message|>", "<|end|>"),
}


@dataclass
class RoleSeq:
    base_id: str
    role: str
    input_ids: list[int]
    content_start: int   # first content-token index (inclusive)
    content_end: int     # last content-token index (exclusive)
    n_content: int
    filler_len: int


def _ids(tok, s: str) -> list[int]:
    return tok.encode(s, add_special_tokens=False)


def build_variants(tok, base_id: str, text: str,
                   max_content_tokens: int = 128,
                   filler_token: str = " the") -> list[RoleSeq]:
    """Build the 5 position-matched role variants for one neutral text.

    Content tokens are truncated to `max_content_tokens` (identical truncation across
    roles, so content is byte-for-byte identical). Content start is aligned across
    roles by left-padding with `filler_token` so every variant places the content at
    the same absolute index (= max prefix length over roles).
    """
    content_ids = _ids(tok, text)[:max_content_tokens]
    filler_id = _ids(tok, filler_token)
    assert len(filler_id) == 1, f"filler {filler_token!r} must be a single token, got {filler_id}"
    filler_id = filler_id[0]

    prefix_ids = {r: _ids(tok, ROLE_WRAP[r][0]) for r in ROLES}
    suffix_ids = {r: _ids(tok, ROLE_WRAP[r][1]) for r in ROLES}
    max_prefix = max(len(prefix_ids[r]) for r in ROLES)

    out = []
    for r in ROLES:
        pad = max_prefix - len(prefix_ids[r])          # leading filler to align content start
        ids = [filler_id] * pad + prefix_ids[r] + content_ids + suffix_ids[r]
        cstart = pad + len(prefix_ids[r])
        cend = cstart + len(content_ids)
        out.append(RoleSeq(base_id, r, ids, cstart, cend, len(content_ids), pad))
    # sanity: content span identical across roles
    starts = {rs.content_start for rs in out}
    ncs = {rs.n_content for rs in out}
    assert len(starts) == 1 and len(ncs) == 1, f"position/length mismatch: {starts} {ncs}"
    return out


START, END, MESSAGE, CHANNEL, RETURN = 200006, 200007, 200008, 200005, 200002


def parse_harmony_spans(ids: list[int], tok) -> list[dict]:
    """Parse a full harmony token sequence into [{role, start, end}] content spans.
    role in {system,user,cot,assistant,tool,developer}. Used for the zero-shot gate."""
    spans, i, n = [], 0, len(ids)
    while i < n:
        if ids[i] == START:
            j = i + 1
            while j < n and ids[j] != MESSAGE and ids[j] != START:
                j += 1
            if j >= n or ids[j] != MESSAGE:
                i += 1; continue
            header = tok.decode(ids[i + 1:j])
            cstart = j + 1
            k = cstart
            while k < n and ids[k] not in (END, RETURN, START):
                k += 1
            if "analysis" in header:
                role = "cot"
            elif "final" in header:
                role = "assistant"
            elif header.strip().startswith("user"):
                role = "user"
            elif header.strip().startswith("system"):
                role = "system"
            elif header.strip().startswith("developer"):
                role = "developer"
            elif "functions" in header or "commentary" in header:
                role = "tool"
            else:
                role = "other"
            if k > cstart and role in ROLES:
                spans.append({"role": role, "start": cstart, "end": k})
            i = k
        else:
            i += 1
    return spans


def build_corpus(tok, neutral_path: str, out_path: str,
                 n_base: int = 250, max_content_tokens: int = 128) -> dict:
    rows = [json.loads(l) for l in open(neutral_path)][:n_base]
    total = 0
    with open(out_path, "w") as f:
        for row in rows:
            for rs in build_variants(tok, row["id"], row["text"], max_content_tokens):
                f.write(json.dumps(asdict(rs)) + "\n")
                total += 1
    return {"n_base": len(rows), "n_seqs": total, "roles": ROLES,
            "max_content_tokens": max_content_tokens}
