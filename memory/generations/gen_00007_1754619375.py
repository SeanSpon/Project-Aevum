def run():
    import random
    score = random.randint(0, 100)
    log = f"Initial brain score: {score}"
    return {"score": score, "log": log}
