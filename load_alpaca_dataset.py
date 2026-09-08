import json

from datasets import load_dataset

OUTPUT_PATH = "indo_input.jsonl"
SAMPLE_SIZE = 20

# Ambil sampel kecil untuk eksperimen, bukan seluruh dataset.
ds = load_dataset("AnnasBlackHat/alpaca-indonesia-llama", split="train")
sample = ds.select(range(min(SAMPLE_SIZE, len(ds))))

# Dataset sumber menyimpan JSON string di dalam field text, jadi unwrap dahulu.
records = [{"text": json.loads(row["text"])["text"]} for row in sample]

with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
    existing = [json.loads(line) for line in f if line.strip()]

seen = {record["text"] for record in existing}
new_records = [record for record in records if record["text"] not in seen]

with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
    for record in new_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Menambahkan {len(new_records)} record ke {OUTPUT_PATH}")
print(f"Total record: {len(existing) + len(new_records)}")
