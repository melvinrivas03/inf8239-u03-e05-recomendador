"""Auditoría de contrato y descriptivos de MovieLens (columnas, claves foráneas, rangos, timestamps)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inf8239_u03 import config  # noqa: E402
from inf8239_u03.data import audit, load_movielens  # noqa: E402

ml = load_movielens()
if config.MANIFEST.exists():
    man = json.loads(config.MANIFEST.read_text())
    print(f"SHA-256 zip: {man['sha256_zip']}  (obtenido {man['retrieved_utc']})")
res = audit(ml)
print(json.dumps(res, indent=2, ensure_ascii=False))
print("Auditoría OK: todas las comprobaciones de contrato pasaron.")
