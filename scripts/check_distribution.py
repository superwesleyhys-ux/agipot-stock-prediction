"""Check current distribution metadata and forbidden archive members."""
from pathlib import Path
import subprocess
import sys
import tarfile
import tomllib
import zipfile

root = Path(__file__).resolve().parents[1]
version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
artifacts = sorted((root / "dist").glob(f"agipot_stock_prediction-{version}*"))
if len(artifacts) != 2:
    raise SystemExit("Expected exactly one wheel and one sdist for the current version")
for path in artifacts:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    else:
        with tarfile.open(path) as archive:
            names = archive.getnames()
    for name in names:
        if any(part in {".env", ".venv", ".harness", ".git", "__pycache__"} for part in Path(name).parts):
            raise SystemExit(f"Unexpected private/runtime path in archive: {name}")
        if Path(name).suffix in {".pem", ".key", ".pyc", ".sqlite", ".db"}:
            raise SystemExit(f"Unexpected runtime/credential file in archive: {name}")
subprocess.run([sys.executable, "-m", "twine", "check", *map(str, artifacts)], check=True)
