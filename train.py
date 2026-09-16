"""Full fine-tune LFM2-350M on data/train.jsonl (prompt/completion, loss on completion only).

    uv run train.py                      # -> out/lfm2-350m-feln
    uv run train.py --check              # print one tokenised row with the loss mask, then exit

Progress is written to <out>/status.json and <out>/metrics.jsonl (same shape as the
feln-liquid runner); `uv run watch.py <out>` is the terminal dashboard for it.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from transformers import TrainerCallback

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

BASE = "LiquidAI/LFM2-350M"  # default; --base LiquidAI/LFM2.5-1.2B-Instruct for the feln-liquid model


class StatusFiles(TrainerCallback):
    """status.json (overwritten) + metrics.jsonl (appended) so a watcher can follow the run."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.started = time.monotonic()
        self.state = {"status": "starting", "started_at": datetime.now(UTC).isoformat()}
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.jsonl").write_text("")

    def _write(self, **values) -> None:
        self.state.update(values, updated_at=datetime.now(UTC).isoformat(), elapsed_seconds=time.monotonic() - self.started)
        tmp = self.out / "status.json.tmp"
        tmp.write_text(json.dumps(self.state, indent=2))
        tmp.replace(self.out / "status.json")

    def on_train_begin(self, args, state, control, **kw):
        self._write(status="training", planned_steps=state.max_steps, epochs=args.num_train_epochs)

    def on_log(self, args, state, control, logs=None, **kw):
        event = {"at": datetime.now(UTC).isoformat(), "step": state.global_step, **(logs or {})}
        with (self.out / "metrics.jsonl").open("a") as fh:
            fh.write(json.dumps(event) + "\n")
        self._write(step=state.global_step, latest_metric=event)

    def on_train_end(self, args, state, control, **kw):
        self._write(status="complete", step=state.global_step)


def open_watcher(out: Path) -> None:
    """Open a Terminal window running watch.py on this run (macOS)."""
    cmd = f"cd {Path(__file__).parent} && uv run watch.py {out}; exit"
    subprocess.run(["osascript", "-e", f'tell application "Terminal" to do script "{cmd}"', "-e", 'tell application "Terminal" to activate'], check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/lfm2-350m-feln")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=768, help="tokens per row; 12288 for the OKF system prompt")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-tui", action="store_true", help="do not open the watch.py Terminal window")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    ds = load_dataset("json", data_files={"train": "data/train.jsonl", "val": "data/val.jsonl"})
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        weight_decay=0.0,
        max_length=args.max_length,
        completion_only_loss=True,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        dataloader_pin_memory=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        processing_class=tok,
        callbacks=[StatusFiles(Path(args.out))],
    )
    if args.check:
        batch = trainer.data_collator([trainer.train_dataset[0]])
        ids, labels = batch["input_ids"][0].tolist(), batch["labels"][0].tolist()
        print("TOTAL", len(ids), "LOSS TOKENS", sum(l != -100 for l in labels))
        print("PROMPT >>>", repr(tok.decode([i for i, l in zip(ids, labels) if l == -100])[-120:]))
        print("TARGET >>>", repr(tok.decode([i for i, l in zip(ids, labels) if l != -100])))
        return
    if not args.no_tui:
        open_watcher(Path(args.out))
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    sys.exit(main())
