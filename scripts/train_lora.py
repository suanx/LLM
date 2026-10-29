"""CPU LoRA SFT for miro-2b: tool-calling reinforcement on the heretic-abliterated MiniCPM5-2B.

Design notes:
- loss only on assistant tokens (labels pre-masked by sbx_prepare_data.py)
- LoRA params kept in fp32 for stable CPU training; base stays bf16
- wall-clock budget so the run always finishes with a usable adapter
"""
import json
import math
import os
import random
import time

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = "/home/daytona/base"
BLOCKS = "/home/daytona/train_blocks.pt"
TPL_PATH = "/home/daytona/chat_template.jinja"
OUT_DIR = "/home/daytona/out"
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "50"))

MAX_STEPS = int(os.environ.get("MAX_STEPS", "800"))
MAX_HOURS = float(os.environ.get("MAX_HOURS", "5.0"))
LR = float(os.environ.get("LR", "1e-4"))
USE_GC = os.environ.get("USE_GC", "1") == "1"
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))


def build_prompt_roundtrip_check(tok, tpl):
    msgs = [{"role": "user", "content": "What's the weather in Beijing tomorrow?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "city name"},
                        "date": {"type": "string", "description": "ISO date"},
                    },
                    "required": ["city"],
                },
            },
        }
    ]
    return tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True), tools, msgs


def generate_sample(model, tok, msgs, tools, max_new_tokens=160):
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(42)
    random.seed(42)

    tok = AutoTokenizer.from_pretrained(BASE_DIR)
    with open(TPL_PATH, encoding="utf-8") as f:
        tok.chat_template = f.read()

    data = torch.load(BLOCKS, weights_only=False)
    blocks = data["blocks"]
    print(f"loaded {len(blocks)} blocks of up to {data['block']} tokens", flush=True)

    print("loading base (bf16, cpu) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_DIR, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    print("params: %.2fB" % (sum(p.numel() for p in model.parameters()) / 1e9), flush=True)

    ckpt_resume = None
    if os.environ.get("RESUME", "0") == "1":
        cands = [d for d in os.listdir(OUT_DIR) if d.startswith("ckpt-")] if os.path.isdir(OUT_DIR) else []
        if cands:
            ckpt_resume = os.path.join(OUT_DIR, sorted(cands, key=lambda x: int(x.split("-")[1]))[-1])

    if ckpt_resume:
        from peft import PeftModel

        print("resuming from", ckpt_resume, flush=True)
        model = PeftModel.from_pretrained(model, ckpt_resume, is_trainable=True)
    else:
        lcfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lcfg)

    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    global USE_GC
    if USE_GC:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    steps = min(MAX_STEPS, len(blocks))
    warmup = min(20, max(1, steps // 10))

    def lr_at(step):
        if step < warmup:
            return LR * (step + 1) / warmup
        prog = (step - warmup) / max(1, steps - warmup)
        return LR * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    order = list(range(len(blocks)))
    random.shuffle(order)

    print(f"training {steps} steps (gc={USE_GC}, lr={LR}, budget={MAX_HOURS}h)", flush=True)
    model.train()
    losses = []
    time_budget = MAX_HOURS * 3600

    for step in range(steps):
        if time.time() - t0 > time_budget:
            print("TIME BUDGET REACHED at step", step, flush=True)
            break
        ids, labels = blocks[order[step % len(order)]]
        input_ids = torch.tensor([ids], dtype=torch.long)
        label_ids = torch.tensor([labels], dtype=torch.long)
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        try:
            out = model(input_ids=input_ids, labels=label_ids)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        except torch.OutOfMemoryError:
            if USE_GC:
                raise
            print("OOM - enabling gradient checkpointing and retrying this step", flush=True)
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            USE_GC = True
            opt.zero_grad(set_to_none=True)
            try:
                out = model(input_ids=input_ids, labels=label_ids)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            except torch.OutOfMemoryError:
                print("OOM persists - stopping early", flush=True)
                break
        losses.append(float(loss.item()))
        if step % 10 == 0 or step == steps - 1:
            el = time.time() - t0
            done = step + 1
            eta = el / done * (steps - done)
            print(
                f"step {done}/{steps} loss={losses[-1]:.4f} lr={lr_at(step):.2e} "
                f"elapsed={el/60:.1f}m eta={eta/60:.1f}m",
                flush=True,
            )
        if (step + 1) % CKPT_EVERY == 0:
            ck = os.path.join(OUT_DIR, f"ckpt-{step+1}")
            model.save_pretrained(ck)
            print("checkpoint ->", ck, flush=True)

    model.save_pretrained(os.path.join(OUT_DIR, "adapter"))
    print("adapter saved ->", os.path.join(OUT_DIR, "adapter"), flush=True)

    with open(os.path.join(OUT_DIR, "loss.json"), "w") as f:
        json.dump({"losses": losses, "steps": len(losses), "elapsed_s": round(time.time() - t0, 1)}, f)

    # ---- sanity check: does the tuned model emit the MiniCPM5 XML tool call?
    prompt, tools, msgs = build_prompt_roundtrip_check(tok, tok.chat_template)
    gen = generate_sample(model, tok, msgs, tools)
    print("=== POST-TRAIN GENERATION ===")
    print(gen[:1200])
    print("=== END GENERATION ===")
    with open(os.path.join(OUT_DIR, "gen_after.txt"), "w", encoding="utf-8") as f:
        f.write(gen)
    print("TRAIN_DONE in %.1f min" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
