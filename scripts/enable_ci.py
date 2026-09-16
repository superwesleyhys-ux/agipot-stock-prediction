"""Copy the reviewed test workflow after the publisher has workflow permission."""
from pathlib import Path
import shutil
root = Path(__file__).resolve().parents[1]
source = root / "docs/harness/workflows/tests.yml"
target = root / ".github/workflows/tests.yml"
if target.exists() and target.read_bytes() != source.read_bytes():
    raise SystemExit("Existing workflow differs; review it without overwriting.")
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(source, target)
print("Prepared .github/workflows/tests.yml; commit and push with workflow permission.")
