"""Extract divan benchmark metrics and output them to json.

Output example:
    [{"name": "benchmark_leiden", "value": 2.3, "unit": "ms"}]
"""

import json
import re
import sys

conversion_table = {
    "m": 60.0,
    "s": 1.0,
    "ms": 1.0e-3,
    "µs": 1.0e-6,
}

matches = re.findall(
    r"(\w+)\s+(\d+(?:\.\d+)?\s+\S+)\s*│\s*(\d+(?:\.\d+)?\s+\S+)\s*│\s*(\d+(?:\.\d+)?\s+\S+)\s*│\s*(\d+(?:\.\d+)?\s+\S+)",
    open(sys.argv[1]).read(),
)

results = [
    {
        "name": m[0],
        "value": float(m[2].split(" ")[0]) * conversion_table[m[2].split(" ")[1]],
        "unit": "s",
    }
    for m in matches
]

print(results)
if not results:
    sys.exit(1)
json.dump(results, open(sys.argv[2], "w"))
