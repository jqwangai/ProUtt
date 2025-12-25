from typing import Any, Dict, List


def messages2history_round(messages: List[Dict[str, Any]]) -> str:
    """
    Convert OpenAI-style message list into a readable multi-round history string.
    Round increases on each new user turn.
    """
    history_lines = []
    round_num = 1
    last_role = None

    for message in messages:
        role = (message.get("role") or "").lower()
        content = (message.get("content") or "").strip()

        if role == "system":
            history_lines.append(f"[System Prompt]: {content}")
            continue

        if role == "user" and last_role != "user":
            history_lines.append(f"\n[Round {round_num}]")
            round_num += 1

        history_lines.append(f"{role.capitalize()}: {content}")
        last_role = role

    return "\n".join(history_lines)


def accumulate_token_usage(total: Dict[str, int], item: Dict[str, int]) -> None:
    total["prompt_tokens"] += item.get("prompt_tokens", 0)
    total["completion_tokens"] += item.get("completion_tokens", 0)
    total["total_tokens"] += item.get("total_tokens", 0)


def top_similarity(result: List[Dict[str, Any]]) -> float:
    """
    result is expected to be a list of dicts with key 'similarity'.
    """
    top_sim = 0.0
    for item in result:
        sim = float(item["similarity"])
        if sim > top_sim:
            top_sim = sim
    return top_sim
