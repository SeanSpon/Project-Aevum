# core/core_logic.py
"""
Tiny stateful learner for Aevum.
Learns y = AX + B (+ noise) with SGD and saves weights across runs.
Crank 'BATCH' and 'STEPS' to make it "think heavier" each generation.
"""

import json, os, random
import numpy as np

# ---- knobs you can tweak ----
BATCH = 32768          # samples per step (heavier thinking => bigger)
STEPS = 3            # SGD steps per generation (keep >1 if you want more work)
TARGET_A = 3.0       # ground-truth slope
TARGET_B = 7.0       # ground-truth intercept
NOISE = 0.1          # label noise amplitude
WARMUP_STEPS = 100   # use boosted LR during warmup
BASE_LR = 0.01       # base learning rate
CLIP = 5.0           # gradient clip (helps stability)
SCORE_SMOOTH = 0.5   # EMA smoothing for score stability, 0..1

STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")

def _load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            try:
                s = json.load(f)
                # add defaults if old state is missing keys
                s.setdefault("score_ema", None)
                s.setdefault("lr", BASE_LR)
                return s
            except Exception:
                pass
    # fresh random init
    return {
        "w": float(random.uniform(-2, 2)),
        "b": float(random.uniform(-1, 1)),
        "lr": float(BASE_LR),
        "step": 0,
        "score_ema": None
    }

def _save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f)

def _make_batch(n=BATCH):
    x = np.random.uniform(-10.0, 10.0, size=(n,)).astype(np.float64)
    y = TARGET_A * x + TARGET_B + np.random.uniform(-NOISE, NOISE, size=(n,)).astype(np.float64)
    return x, y

def _sgd_step(w, b, lr):
    x, y = _make_batch()
    yhat = w * x + b
    err = (yhat - y)
    loss = np.mean(err * err)

    # gradients for MSE
    dw = 2.0 * np.mean(err * x)
    db = 2.0 * np.mean(err)

    # clip for stability
    gnorm = np.sqrt(dw * dw + db * db)
    if gnorm > CLIP:
        scale = CLIP / (gnorm + 1e-12)
        dw *= scale
        db *= scale

    # update
    w -= lr * dw
    b -= lr * db

    return w, b, float(loss)

def run():
    s = _load_state()
    w, b, lr, step, score_ema = s["w"], s["b"], s["lr"], s["step"], s["score_ema"]

    # warmup → boosted LR; then cosine decay toward base LR
    if step < WARMUP_STEPS:
        lr_use = lr * 5.0
    else:
        # gentle decay factor after warmup
        t = min(1.0, (step - WARMUP_STEPS) / 1000.0)
        lr_use = lr * (0.5 * (1.0 + np.cos(np.pi * t)))

    total_loss = 0.0
    w_new, b_new = w, b
    for _ in range(max(1, int(STEPS))):
        w_new, b_new, loss = _sgd_step(w_new, b_new, lr_use)
        total_loss += loss
    avg_loss = total_loss / max(1, int(STEPS))

    # score: higher when loss is lower; smoothed for stability
    raw_score = max(0.0, min(100.0, 100.0 / (1.0 + avg_loss)))
    if score_ema is None:
        score = raw_score
    else:
        score = (1.0 - SCORE_SMOOTH) * score_ema + SCORE_SMOOTH * raw_score

    step += 1
    s.update({
        "w": float(w_new),
        "b": float(b_new),
        "lr": float(lr),
        "step": int(step),
        "score_ema": float(score)
    })
    _save_state(s)

    log = (
        f"step={step} loss={avg_loss:.6f} "
        f"w={w_new:.4f} b={b_new:.4f} lr_used={lr_use:.5f} "
        f"batch={BATCH} steps={STEPS}"
    )
    return {"score": float(score), "log": log}