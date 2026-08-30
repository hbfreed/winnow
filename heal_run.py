"""Heal a pruned Trinity checkpoint with LoRA adapters on the ragged runtime.

Trains on tulu-3 chat-templated sequences disjoint from both the calibration
slice (rows 0-1023) and the held-out evaluation slice (rows 1024-1087), then
merges the adapters and writes a standard Winnow checkpoint.
"""

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from winnow.calibration import calibration_batches
from winnow.heal import attach_healing, merge_healing

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--accumulation", type=int, default=8)
parser.add_argument("--sequence-length", type=int, default=2048)
parser.add_argument("--rank", type=int, default=16)
parser.add_argument("--alpha", type=float, default=32.0)
parser.add_argument("--lora-lr", type=float, default=1e-4)
parser.add_argument("--router-lr", type=float, default=5e-6)
parser.add_argument("--eval-every", type=int, default=100)
parser.add_argument("--skip-sequences", type=int, default=1088, help="calibration+eval rows")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.checkpoint, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto"
)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.train()
trainable = attach_healing(model, rank=args.rank, alpha=args.alpha)
lora, router = [], []
for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        (router if "router" in name or ".gate.weight" in name else lora).append(parameter)
print(
    f"trainable: {len(trainable)} tensors, {sum(p.numel() for p in lora + router) / 1e6:.1f}M params",
    flush=True,
)
optimizer = torch.optim.AdamW(
    [{"params": lora, "lr": args.lora_lr}, {"params": router, "lr": args.router_lr}],
    weight_decay=0.0,
)
schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
input_device = model.get_input_embeddings().weight.device


def _rows(skip: int, count: int):
    rows = []
    for batch in calibration_batches(
        tokenizer,
        "allenai/tulu-3-sft-mixture",
        text_field="messages",
        chat_template=True,
        sequences=skip + count,
        sequence_length=args.sequence_length,
        batch_size=1,
        seed=0,
    ):
        rows.append(batch[0])
    return rows[skip:]


held_out = torch.stack(_rows(1024, 64))  # the exact slice every prior number used


@torch.no_grad()
def evaluate() -> float:
    model.eval()
    loss_sum, tokens = 0.0, 0
    for row in held_out:
        ids = row.unsqueeze(0).to(input_device)
        logits = model(ids, use_cache=False).logits[0, :-1].float()
        loss_sum += float(F.cross_entropy(logits, ids[0, 1:].to(logits.device), reduction="sum"))
        tokens += ids.shape[1] - 1
    model.train()
    return math.exp(loss_sum / tokens)


print(f"pre-heal held-out perplexity {evaluate():.4f}", flush=True)

train_rows = iter(_rows(args.skip_sequences, args.steps * args.accumulation))
started = time.time()
for step in range(1, args.steps + 1):
    optimizer.zero_grad(set_to_none=True)
    accumulated = 0.0
    for _micro in range(args.accumulation):
        ids = next(train_rows).unsqueeze(0).to(input_device)
        logits = model(ids, use_cache=False).logits[0, :-1].float()
        loss = F.cross_entropy(logits, ids[0, 1:].to(logits.device)) / args.accumulation
        loss.backward()
        accumulated += float(loss)
    torch.nn.utils.clip_grad_norm_(lora + router, 1.0)
    optimizer.step()
    schedule.step()
    if step % 20 == 0:
        pace = (time.time() - started) / step
        print(f"step {step}/{args.steps} loss {accumulated:.4f} ({pace:.1f}s/step)", flush=True)
    if step % args.eval_every == 0:
        print(f"step {step} held-out perplexity {evaluate():.4f}", flush=True)

final_ppl = evaluate()
print(f"post-heal held-out perplexity {final_ppl:.4f}", flush=True)

merge_healing(model)
model.save_pretrained(args.output, safe_serialization=True)
tokenizer.save_pretrained(args.output)
for support in (
    "winnow.json",
    "generation_config.json",
    "configuration_winnow_afmoe.py",
    "modeling_winnow_afmoe.py",
):
    source_file = Path(args.checkpoint) / support
    if source_file.exists():
        shutil.copyfile(source_file, args.output / support)
manifest = json.loads((args.output / "winnow.json").read_text())
manifest["healing"] = {
    "method": "lora (per-expert + attention) + full router gate",
    "rank": args.rank,
    "alpha": args.alpha,
    "steps": args.steps,
    "tokens": args.steps * args.accumulation * args.sequence_length,
    "dataset": "allenai/tulu-3-sft-mixture (chat template, seed 0, rows disjoint from calibration/eval)",
    "lora_lr": args.lora_lr,
    "router_lr": args.router_lr,
    "post_heal_perplexity": final_ppl,
}
(args.output / "winnow.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(f"Wrote {args.output}", flush=True)
