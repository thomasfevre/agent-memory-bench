import json
from pathlib import Path


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8"))["completed"])


def save_completed(path: Path, completed: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"completed": sorted(completed)}) + "\n",
        encoding="utf-8",
    )
