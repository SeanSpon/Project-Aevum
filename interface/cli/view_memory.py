import json, os

MEMORY_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "memory", "memory.json"))

def main():
    if not os.path.exists(MEMORY_FILE):
        print("No memory.json found yet. Run Project_Aevum.py first.")
        return
    with open(MEMORY_FILE, "r") as f:
        data = json.load(f)
    print(f"Entries: {len(data)}")
    for i, entry in enumerate(data[-25:], 1):
        t = entry.get("time", "?")
        ev = entry.get("event", "run")
        score = entry.get("score", "-")
        log = entry.get("log", "")
        mutated = entry.get("mutated", False)
        print(f"{i:02d}. {t} | event={ev} | score={score} | mutated={mutated} | {log}")

if __name__ == "__main__":
    main()
