import json
import os

input_path = "data/GSM8K_zh/GSM8K_zh_train.json"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

train_data = [item for item in data if item.get("split") == "train"]
test_data = [item for item in data if item.get("split") == "test"]

with open(os.path.join(output_dir, "GSM8K_zh_train.json"), "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(os.path.join(output_dir, "GSM8K_zh_test.json"), "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print("Done!")