"""Reference for the MTP draft step (docs/mtp.md) on the tiny checkpoint: tests/fixtures/mtp/manifest.json.

Implements the vLLM `Qwen3NextMultiTokenPredictor` op order in PyTorch on the HF side of the weights:
`fc([pre_fc_norm_embedding(embed(x)) | pre_fc_norm_hidden(h)])` -> one full-attention `Qwen3_5DecoderLayer`
(its own KV cache) -> `norm` -> the shared `lm_head`; the chained row takes the draft token and the MTP's own
post-`norm` hidden. `h` is the target's post-final-norm hidden (`model.model.norm` output), as vLLM / SGLang /
llama.cpp all serve it. Norms are `Qwen3_5RMSNorm` ((1 + w) * normed, zero-centered), which the GGUF stores
with the +1 folded in.

Per fixture prompt (tests/fixtures/prompts.json): the target model runs the prompt (T ids) and greedily emits
x_T; the MTP consumes rows (x_{j+1}, h_j) for j = 0..T-1 at positions 0..T-1 (row T-1 pairs x_T with h_{T-1});
draft 1 is the argmax of the last row; drafts 2..20 chain. Filed: draft-1 logits (full vocab), the 20 draft ids,
top-5 ids / logits and the top-1/top-2 margin per step, max|logits| per step.

Usage: python tools/ref_mtp.py [--model models/tiny] [--drafts 20]
"""
import argparse
import copy
import json
import os
import sys

import torch
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "fixtures"))
from common import Fixture  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import (  # noqa: E402
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)


class MtpHead(torch.nn.Module):
    """vLLM Qwen3NextMultiTokenPredictor, HF modules, float32."""

    def __init__(self, cfg, sd):
        super().__init__()
        h = cfg.hidden_size
        self.fc = torch.nn.Linear(2 * h, h, bias=False)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(h, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(h, eps=cfg.rms_norm_eps)
        self.norm = Qwen3_5RMSNorm(h, eps=cfg.rms_norm_eps)
        # one decoder block of type full_attention: a config copy whose layer 0 is full attention
        lcfg = copy.deepcopy(cfg)
        lcfg.layer_types = ["full_attention"] + list(cfg.layer_types[1:])
        lcfg._attn_implementation = "eager"
        self.layer = Qwen3_5DecoderLayer(lcfg, layer_idx=0)
        self.rotary = Qwen3_5TextRotaryEmbedding(config=lcfg)
        self.lcfg = lcfg
        with torch.no_grad():
            self.fc.weight.copy_(sd["mtp.fc.weight"].float())
            self.pre_fc_norm_embedding.weight.copy_(sd["mtp.pre_fc_norm_embedding.weight"].float())
            self.pre_fc_norm_hidden.weight.copy_(sd["mtp.pre_fc_norm_hidden.weight"].float())
            self.norm.weight.copy_(sd["mtp.norm.weight"].float())
            lsd = {k[len("mtp.layers.0."):]: v.float() for k, v in sd.items() if k.startswith("mtp.layers.0.")}
            missing, unexpected = self.layer.load_state_dict(lsd, strict=False)
            assert not unexpected, unexpected
            assert not missing, missing
        self.eval()

    @torch.no_grad()
    def rows(self, embeds, hiddens, positions, cache):
        """`n` rows: token embeddings (n, h), target hiddens (n, h), positions (n,). Returns the post-norm
        hidden (n, h) for every row (row-wise the same as feeding them one at a time)."""
        e = self.pre_fc_norm_embedding(embeds)
        hn = self.pre_fc_norm_hidden(hiddens)
        u = self.fc(torch.cat([e, hn], dim=-1)).unsqueeze(0)  # (1, n, h)
        pos = positions.view(1, -1)
        pe = self.rotary(u, pos)  # 2-D position ids: the module expands them to the 3 equal mRoPE streams
        n = u.shape[1]
        past = cache.get_seq_length(0) if cache is not None else 0
        # causal mask over past + n keys, float additive
        total = past + n
        mask = torch.full((1, 1, n, total), torch.finfo(torch.float32).min)
        for i in range(n):
            mask[0, 0, i, : past + i + 1] = 0.0
        out = self.layer(u, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True,
                         cache_position=torch.arange(past, past + n), position_embeddings=pe)
        out = out[0] if isinstance(out, tuple) else out
        return self.norm(out)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "tiny"))
    ap.add_argument("--drafts", type=int, default=20)
    a = ap.parse_args()
    model = Qwen3_5ForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="eager").eval()
    cfg = model.config
    sd = load_file(os.path.join(a.model, "model.safetensors"))
    mtp = MtpHead(cfg, sd)
    embed = model.model.embed_tokens
    lm_head = model.lm_head
    prompts = json.load(open(os.path.join(ROOT, "tests", "fixtures", "prompts.json")))["prompts"]
    fx = Fixture("mtp", subdir="mtp")
    fx.meta["source"] = "tools/ref_mtp.py: vLLM Qwen3NextMultiTokenPredictor op order (docs/mtp.md) with HF modules on models/tiny, float32, eager"
    fx.meta["drafts"] = a.drafts
    fx.meta["budget_k_per_layer"] = 16
    fx.meta["budget_note"] = "draft logits: k * (n_layer + 2 + 1) * sqrt(hidden) * eps * max|logits| (the target's budget plus one block)"
    fx.meta["hidden_is"] = "target post-final-norm hidden (model.model.norm output); chained rows use the MTP post-norm hidden"
    fx.meta["row_convention"] = "row j = (x_{j+1}, h_j) at position j; the prompt's T rows are fed as one batch, draft 1 from the last"
    captured = {}
    hook = model.model.norm.register_forward_hook(lambda m, i, o: captured.__setitem__("norm", o.detach().clone()))
    for p in prompts:
        ids = torch.tensor([p["ids"]], dtype=torch.long)
        t = ids.shape[1]
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
        h = captured["norm"][0]  # (T, hidden): post-final-norm at every prompt position
        x_t = int(out.logits[0, -1].argmax())
        toks = torch.tensor(p["ids"][1:] + [x_t], dtype=torch.long)  # x_1 .. x_T
        cache = DynamicCache(config=mtp.lcfg)
        with torch.no_grad():
            hp = mtp.rows(embed(toks), h, torch.arange(t), cache)
            logits = lm_head(hp[-1]).float()
        draft_logits_1 = logits.clone()
        drafts, top5_ids, top5_vals, margins, max_abs = [], [], [], [], []
        cur = logits
        hcur = hp[-1]
        pos = t
        with torch.no_grad():
            for step in range(a.drafts):
                d = int(cur.argmax())
                top = torch.topk(cur, 5)
                drafts.append(d)
                top5_ids.append([int(i) for i in top.indices])
                top5_vals.append([float(v) for v in top.values])
                margins.append(float(top.values[0] - top.values[1]))
                max_abs.append(float(cur.abs().max()))
                if step + 1 == a.drafts:
                    break
                hcur = mtp.rows(embed(torch.tensor([d])), hcur.view(1, -1), torch.tensor([pos]), cache)[0]
                cur = lm_head(hcur).float()
                pos += 1
        c = p["name"]
        fx.add(c, prompt_ids=p["ids"], n_ids=t, first_target_token=x_t, draft_logits_1=fx.array(c, "draft_logits_1", draft_logits_1),
               draft_logits_1_max_abs=float(draft_logits_1.abs().max()), draft_ids=drafts, top5_ids=top5_ids, top5_logits=top5_vals,
               margins=margins, max_abs=max_abs, mtp_rows_after=int(cache.get_seq_length(0)))
        print("%-9s T=%d x_T=%d drafts=%s min margin=%.4f" % (c, t, x_t, drafts, min(margins)))
    hook.remove()
    fx.write()


if __name__ == "__main__":
    main()
