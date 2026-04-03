import json
with open("result.json") as f:
    data = json.load(f)

for idx, sample in enumerate(data["answers"]):
    print(f"Sample {idx}: {sample['prediction'][:50]}")
