"""CPU LoRA PoC: prove the tool-calling SFT pipeline on the heretic-abliterated MiniCPM5-2B.
- loads the abliterated base (bf16, cpu)
- applies a LoRA adapter
- trains a few steps on a small slice of the openbmb agent tool-calling data
- reports the loss curve, saves the adapter, and generates a sample tool-calling turn
"""
import os, json, time
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

BASE_REPO = "insraq/MiniCPM5-2B-heretic-abliterated"
DATA = "/home/daytona/agent_general.jsonl"
N_SAMPLES = int(os.environ.get("POC_N", "150"))     # rows to pull
STEPS = int(os.environ.get("POC_STEPS", "12"))       # CPU PoC steps
CUTOFF = int(os.environ.get("POC_CUTOFF", "768"))
LORA_R, LORA_ALPHA = 16, 32

def main():
    import torch as _t
    _t.set_num_threads(4)   # sandbox is 4-core (host reports more)
    t0 = time.time()
    base_dir = snapshot_download(BASE_REPO)  # cached, no re-download
    print("BASE_DIR=", base_dir, flush=True)

    import transformers
    print("transformers", transformers.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
    tok = AutoTokenizer.from_pretrained(base_dir)
    print("tokenizer ok; eos", tok.eos_token, "cls", type(tok).__name__, flush=True)

    # standard Llama architecture -> no trust_remote_code needed
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.gradient_checkpointing_enable()
    print("model loaded (bf16, cpu). params: %.2fB" % (sum(p.numel() for p in model.parameters())/1e9), flush=True)

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # build a tiny dataset
    rows = []
    with open(DATA, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= N_SAMPLES: break
            try:
                rec = json.loads(line)
            except Exception:
                continue
            msgs = rec.get("messages"); tools = rec.get("tools") or []
            if not msgs: continue
            rows.append((msgs, tools))
    print("usable samples:", len(rows), flush=True)

    def mk_tensor(msgs, tools):
        text = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=False, tokenize=False)
        inp = tok(text, return_tensors="pt", truncation=True, max_length=CUTOFF)
        return inp["input_ids"], inp["attention_mask"]

    # training loop (full-sequence loss for the PoC; production uses prompt masking)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    losses = []
    for step in range(STEPS):
        msgs, tools = rows[step % len(rows)]
        ids, mask = mk_tensor(msgs, tools)
        ids, mask = ids.to("cpu"), mask.to("cpu")
        out = model(input_ids=ids, attention_mask=mask, labels=ids)
        loss = out.loss
        loss.backward()
        opt.step(); opt.zero_grad()
        losses.append(float(loss.item()))
        if step % 3 == 0 or step == STEPS - 1:
            print(f"step {step+1}/{STEPS} loss={losses[-1]:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # save adapter
    os.makedirs("/home/daytona/lora_adapter", exist_ok=True)
    model.save_pretrained("/home/daytona/lora_adapter")
    print("adapter saved -> /home/daytona/lora_adapter", flush=True)
    print("LOSS_CURVE=", json.dumps(losses), flush=True)

    # generate a sample tool-calling turn with the LoRA
    model.eval()
    demo_msgs = rows[0][0]; demo_tools = rows[0][1]
    demo_text = tok.apply_chat_template(demo_msgs[:-1], tools=demo_tools, add_generation_prompt=True, tokenize=False)
    sample_out = ""
    with torch.no_grad():
        enc = tok(demo_text, return_tensors="pt", truncation=True, max_length=CUTOFF)
        gen = model.generate(**enc, max_new_tokens=256, do_sample=False)
        sample_out = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)[:1500]
    print("=== SAMPLE GENERATION (LoRA) ===")
    print(sample_out)

    # persist a compact result for retrieval
    with open("/home/daytona/poc_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "base": BASE_REPO, "n_samples": N_SAMPLES, "steps": STEPS, "cutoff": CUTOFF,
            "lora": {"r": LORA_R, "alpha": LORA_ALPHA},
            "loss_curve": losses,
            "elapsed_s": round(time.time() - t0, 1),
            "sample_generation": sample_out,
        }, f, ensure_ascii=False, indent=2)
    print("result -> /home/daytona/poc_result.json", flush=True)
    print("POC_DONE", flush=True)

if __name__ == "__main__":
    main()
