"""Build an immutable seed exclusively from existing independent train records."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

source, output = map(Path, sys.argv[1:3])
counts = Counter()
hashes = []
for path in sorted(source.glob("*.json")):
    item = json.loads(path.read_text())
    if item["source_split"] != "train" or item["split"] != "train":
        continue
    counts.update(item["prompt_ids"])
    counts.update(item["generated_ids"])
    hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
assert len(hashes) == 64
ids = [i for i, _ in counts.most_common(1024)]
# All model-defined special IDs; no need to assume tokenizer ID ranges.
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(sys.argv[3], local_files_only=True)
ids = list(dict.fromkeys(ids + tokenizer.all_special_ids))
result = {"token_ids": ids, "source": "64 previously generated independent GSM8K train examples",
          "source_record_sha256": hashes, "records": len(hashes),
          "observations": sum(counts.values()), "test_used": False}
output.write_text(json.dumps(result, indent=2))
print({k:v for k,v in result.items() if k not in ("token_ids", "source_record_sha256")})
