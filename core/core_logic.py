# core/core_logic.py
import json, os, random

STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")

def _load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"w": random.uniform(-2, 2), "b": random.uniform(-1, 1), "lr": 0.01, "step": 0}

def _save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f)

def _batch(n=128):
    xs, ys = [], []
    for _ in range(n):
        x = random.uniform(-10, 10)
        y = 3.0 * x + 7.0 + random.uniform(-0.2, 0.2)  # target rule + noise
        xs.append(x); ys.append(y)
    return xs, ys

def run():
    s = _load_state()
    w, b, lr, step = s["w"], s["b"], s["lr"], s["step"]

    xs, ys = _batch()
    dw = db = loss = 0.0
    for x, y in zip(xs, ys):
        yhat = w * x + b
        err = yhat - y
        loss += err*err
        dw += 2.0 * err * x
        db += 2.0 * err
    n = float(len(xs))
    loss /= n; dw /= n; db /= n

    lr_use = lr * 5.0 if step < 50 else lr
    w -= lr_use * dw
    b -= lr_use * db
    step += 1

    score = max(0.0, min(100.0, 100.0 / (1.0 + loss)))
    s.update({"w": w, "b": b, "lr": lr, "step": step})
    _save_state(s)

    log = f"step={step} loss={loss:.4f} w={w:.3f} b={b:.3f} lr={lr_use:.4f}"
    return {"score": score, "log": log}
