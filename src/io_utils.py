import json
from typing import Any, Dict, Iterable, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    """
    Append one JSON object as a line.
    Note: called from multiple async tasks; use an async-safe writer if needed.
    In practice, a single-process append often works, but it can interleave under heavy concurrency.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def sum_usage_from_jsonl(path: str) -> Dict[str, int]:
    usage_all = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            usage = item.get("usage", {})
            usage_all["prompt_tokens"] += usage.get("prompt_tokens", 0)
            usage_all["completion_tokens"] += usage.get("completion_tokens", 0)
            usage_all["total_tokens"] += usage.get("total_tokens", 0)
    return usage_all
