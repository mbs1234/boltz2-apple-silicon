"""Generate one MSA per construct, once, with boltz's own compute_msa (ColabFold server)."""
import os
import datetime, hashlib, json
from pathlib import Path
from boltz.main import compute_msa

HOME = Path(os.environ.get("BOLTZ2_HOME", Path.home() / "boltz2-dev"))
ROOT = HOME / "inputs"
URL, PAIRING = "https://api.colabfold.com", "greedy"
seqs, k = {}, None
for line in open(ROOT / "constructs.fasta"):
    line = line.strip()
    if line.startswith(">"):
        k = line[1:].split()[0]; seqs[k] = ""
    elif line:
        seqs[k] += line
rec = {"server": URL, "pairing": PAIRING,
       "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "msas": {}}
for name, seq in seqs.items():
    tid = name.replace("-", "_")
    compute_msa({f"{tid}_0": seq}, tid, ROOT / "msa", URL, PAIRING)
    out = ROOT / "msa" / f"{tid}_0.csv"
    rows = sum(1 for _ in open(out)) - 1
    rec["msas"][name] = {"file": out.name, "rows": rows,
                         "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
(HOME / "provenance" / "msas.json").write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1))
