"""Terminal dashboard for a train.py run: reads <out>/status.json + metrics.jsonl every 2 s.

    uv run watch.py out/lfm2-350m-feln
"""

import json
import sys
import time
from pathlib import Path

BAR = 40


def render(out: Path) -> str:
    status = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
    rows = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines() if l.strip()] if (out / "metrics.jsonl").exists() else []
    step, total = status.get("step", 0), status.get("planned_steps") or 1
    frac = min(step / total, 1.0)
    elapsed = status.get("elapsed_seconds", 0.0)
    eta = elapsed / frac - elapsed if frac > 0 else float("nan")
    evals = [r for r in rows if "eval_loss" in r]
    last_eval = f"   eval loss {evals[-1]['eval_loss']:.4f} (epoch {evals[-1].get('epoch', 0):.0f})" if evals else "   eval loss: at end of each epoch"
    fmt = lambda r, k, spec: format(r[k], spec) if k in r else ""
    lines = [
        f"\033[1m{out}\033[0m   status: {status.get('status', '?')}{last_eval}",
        f"[{'#' * int(frac * BAR):<{BAR}}] {step}/{total}  {frac:6.1%}   elapsed {elapsed / 60:5.1f} min   eta {eta / 60:5.1f} min",
        "",
        f"{'step':>6} {'epoch':>6} {'loss':>9} {'lr':>10} {'grad':>7} {'tok_acc':>8} {'eval_loss':>10}",
    ]
    for r in rows[-18:]:
        lines.append(
            f"{r.get('step', ''):>6} {fmt(r, 'epoch', '.2f'):>6} {fmt(r, 'loss', '.4f'):>9} "
            f"{fmt(r, 'learning_rate', '.2e'):>10} {fmt(r, 'grad_norm', '.2f'):>7} "
            f"{fmt(r, 'mean_token_accuracy', '.4f'):>8} {fmt(r, 'eval_loss', '.4f'):>10}"
        )
    return "\n".join(lines)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/lfm2-350m-feln")
    while True:
        print("\033[2J\033[H" + render(out), flush=True)
        if json.loads((out / "status.json").read_text()).get("status") == "complete" if (out / "status.json").exists() else False:
            print("\ntraining complete")
            return
        time.sleep(2)


if __name__ == "__main__":
    main()
