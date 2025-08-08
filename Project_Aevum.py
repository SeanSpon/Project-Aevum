import os, json, time, importlib, shutil, datetime

MEMORY_FILE = "memory/memory.json"
CORE_MODULE = "core.core_logic"
GENERATIONS_DIR = "memory/generations"

def _now_iso():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def ensure_memory():
    if not os.path.exists("memory"):
        os.makedirs("memory", exist_ok=True)
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "w") as f:
            json.dump([], f)

def log_event(entry):
    ensure_memory()
    data = []
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    entry["time"] = entry.get("time", _now_iso())
    data.append(entry)
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

def snapshot_generation(generation:int, note:str=""):
    os.makedirs(GENERATIONS_DIR, exist_ok=True)
    tag = f"gen_{generation:05d}_{int(time.time())}"
    dst = os.path.join(GENERATIONS_DIR, tag + ".py")
    shutil.copyfile("core/core_logic.py", dst)
    meta = {
        "generation": generation,
        "tag": tag,
        "note": note,
        "time": _now_iso()
    }
    with open(os.path.join(GENERATIONS_DIR, tag + ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    return tag

def run_core():
    try:
        if CORE_MODULE in list(importlib.sys.modules.keys()):
            importlib.reload(importlib.import_module(CORE_MODULE))
        core = importlib.import_module(CORE_MODULE)
        result = core.run()
        score = float(result.get("score", 0))
        log_event({
            "event": "run",
            "score": score,
            "log": result.get("log", ""),
            "mutated": False
        })
        return score, result.get("log", "")
    except Exception as e:
        log_event({
            "event": "error",
            "score": 0.0,
            "log": f"{type(e).__name__}: {e}",
            "mutated": False
        })
        return 0.0, str(e)

def mutate_core(strategy: str = "random_curve"):
    # Writes a new brain into core/core_logic.py
    code = (
        "def run():\n"
        "    import random, math\n"
        f"    # Strategy: {strategy}\n"
        "    base = random.randint(0, 100)\n"
        "    bonus = 0\n"
        "    if base < 30:\n"
        "        bonus = int(math.sqrt(base) * 3)\n"
        "    elif base > 70:\n"
        "        bonus = int(math.log(max(base,1)) * 2)\n"
        "    score = max(0, min(100, base + bonus))\n"
        f"    log = 'Strategy={strategy} base={{base}} bonus={{bonus}} -> score={{score}}'.format(base=base, bonus=bonus, score=score)\n"
        "    return {\"score\": score, \"log\": log}\n"
    )
    with open("core/core_logic.py", "w", encoding="utf-8") as f:
        f.write(code)

def main(threshold:float=30.0, sleep_seconds:float=2.5):
    ensure_memory()
    generation = 1
    print("💡 Aevum starting… (Ctrl+C to stop)")
    while True:
        print(f"[Generation {generation}] Running core…")
        score, log = run_core()
        tag = snapshot_generation(generation, note=f"score={score}")
        print(f"  → Score={score:.2f} | Snapshot={tag}")
        if score < threshold:
            print("  ⚠️ Low score: mutating brain…")
            log_event({"event": "mutate", "reason": f"score<{threshold}", "mutated": True})
            mutate_core(strategy="random_curve")
        else:
            print("  ✅ Satisfactory performance. No mutation.")
        time.sleep(sleep_seconds)
        generation += 1

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Aevum halted by user.")
