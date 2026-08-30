"""Minimal OpenAI-compatible chat server for fused Winnow checkpoints.

Static batching: concurrent requests are collected for a short window and
generated together, which is what openbench needs to get throughput out of a
single fused model instance. Reasoning (`<think>...</think>`) is stripped from
the returned content, mirroring vllm's deepseek_r1 parser for Trinity models.
"""

import argparse
import queue
import threading
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from transformers import AutoTokenizer

from winnow.runtime.load import load_fast

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--port", type=int, default=8555)
parser.add_argument("--max-batch", type=int, default=8)
parser.add_argument("--max-new-tokens", type=int, default=6144)
parser.add_argument("--temperature", type=float, default=0.15)
parser.add_argument("--top-p", type=float, default=0.75)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--min-p", type=float, default=0.06)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = load_fast(args.checkpoint)
print(f"serving {args.checkpoint} on port {args.port}", flush=True)

pending: queue.Queue = queue.Queue()


def _strip_reasoning(text: str) -> str:
    marker = "</think>"
    if marker in text:
        return text.split(marker, 1)[1].lstrip("\n")
    return text


def _worker() -> None:
    while True:
        batch = [pending.get()]
        deadline = time.time() + 0.2
        while len(batch) < args.max_batch:
            try:
                batch.append(pending.get(timeout=max(0.0, deadline - time.time())))
            except queue.Empty:
                break
        prompts = [item["prompt"] for item in batch]
        max_new = min(args.max_new_tokens, max(item["max_tokens"] for item in batch))
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        try:
            with torch.no_grad():
                output = model.generate(
                    encoded["input_ids"].to("cuda:0"),
                    attention_mask=encoded["attention_mask"].to("cuda:0"),
                    max_new_tokens=max_new,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature or None,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
                    pad_token_id=tokenizer.pad_token_id,
                    # Trinity ships no eos in its generation config; without an
                    # explicit end-of-turn id, generate never stops.
                    eos_token_id=tokenizer.eos_token_id,
                )
            width = encoded["input_ids"].shape[1]
            for item, row in zip(batch, output):
                completion = row[width:]
                ends = (completion == tokenizer.eos_token_id).nonzero()
                stop = ends.numel() > 0
                if stop:  # trim batch padding so token accounting is honest
                    completion = completion[: int(ends[0]) + 1]
                text = tokenizer.decode(completion, skip_special_tokens=True)
                item["result"].put(
                    {
                        "text": _strip_reasoning(text),
                        "prompt_tokens": width,
                        "completion_tokens": int(completion.numel()),
                        "finish_reason": "stop" if stop else "length",
                    }
                )
        except Exception as error:  # noqa: BLE001 — every waiter must be released
            for item in batch:
                item["result"].put({"error": str(error)})


threading.Thread(target=_worker, daemon=True).start()

app = FastAPI()


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": args.checkpoint, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat(body: dict):
    prompt = tokenizer.apply_chat_template(
        body["messages"], add_generation_prompt=True, tokenize=False
    )
    result: queue.Queue = queue.Queue()
    pending.put(
        {
            "prompt": prompt,
            "max_tokens": int(
                body.get("max_tokens") or body.get("max_completion_tokens") or args.max_new_tokens
            ),
            "result": result,
        }
    )
    outcome = result.get()
    if "error" in outcome:
        return {"error": {"message": outcome["error"], "type": "server_error"}}
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": args.checkpoint,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": outcome["text"]},
                "finish_reason": outcome["finish_reason"],
            }
        ],
        "usage": {
            "prompt_tokens": outcome["prompt_tokens"],
            "completion_tokens": outcome["completion_tokens"],
            "total_tokens": outcome["prompt_tokens"] + outcome["completion_tokens"],
        },
    }


uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
