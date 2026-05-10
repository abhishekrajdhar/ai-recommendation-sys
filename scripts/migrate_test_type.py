import json
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[1] / "app" / "data" / "shl_catalog.json"

mapping = {
    "P": "Personality",
    "A": "Cognitive",
    "K": "Technical",
    "S": "Situational Judgment",
    "O": "General",
}


def migrate():
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    changed = 0
    for item in data:
        tt = item.get("test_type")
        if tt in mapping:
            item["test_type"] = mapping[tt]
            changed += 1
    if changed:
        CATALOG.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Migrated {changed} records")


if __name__ == '__main__':
    migrate()
