# postprocess.py
# Convert synthesized_raw JSONL into:
#   (1) SFT format: {"messages":[{"role":"user",...},{"role":"assistant",...}]}
#   (2) Preference format: {"conversations":[...],"chosen":...,"rejected":...}
#

import argparse
import json
from typing import Any, Dict, List

from utils import messages2history_round
from prompts import INFER_PROMPT, OUTPUT_PROMPT  # keep long prompts out of this file


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_json(path: str, obj: Any) -> None:
    """Write a JSON file with indentation."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def get_item_id(item: Dict[str, Any]) -> str:
    """Best-effort ID for logging."""
    return str(item.get("id", item.get("sample_id", "unknown")))


def build_user_content(context_messages: List[Dict[str, Any]]) -> str:
    """Construct the user-side instruction by filling chat history into INFER_PROMPT."""
    chat_history = messages2history_round(context_messages)
    return INFER_PROMPT.format(chat_history=chat_history)


def extract_predictions(block: Dict[str, Any]) -> List[str]:
    """Concatenate mining-view and exploration-view predictions."""
    return (
        block["insight_reason"]["mining_view"]["predictions"]
        + block["insight_reason"]["explore_view"]["predictions"]
    )


def wrap_predictions(preds: List[str], item_id: str) -> str:
    """
    Wrap predictions using <predict> tags.
    Also warns if prompt artifacts leak into predictions.
    """
    lines: List[str] = []
    for pred in preds:
        if "intent tree" in pred.lower() or "intent_tree" in pred.lower():
            print(f"[Warning] item {item_id}: possible artifact in prediction: {pred[:80]}...")
        lines.append(f"<predict>{pred}</predict>")
    return "\n".join(lines) + ("\n" if lines else "")


def format_assistant_output(block: Dict[str, Any], predictions_wrapped: str) -> str:
    """Fill OUTPUT_PROMPT using one block (chosen or rejected)."""
    return OUTPUT_PROMPT.format(
        intent_tree=block["intent_tree"],
        utterance_category_reasoning=block["utterance_category_reason"]["reasoning"],
        utterance_category=block["utterance_category_reason"]["category"],
        mining_reasoning=block["insight_reason"]["mining_view"]["reasoning"],
        exploration_reasoning=block["insight_reason"]["explore_view"]["reasoning"],
        predictions=predictions_wrapped,
    )


def convert_to_sft(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    SFT positive example: uses ONLY the chosen branch as the assistant output.
    """
    item_id = get_item_id(item)
    user_content = build_user_content(item["context"])

    chosen_block = item["chosen"]
    preds_wrapped = wrap_predictions(extract_predictions(chosen_block), item_id)
    assistant_content = format_assistant_output(chosen_block, preds_wrapped)

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def convert_to_pref(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preference (DPO-style) example: one human prompt with a chosen/rejected assistant response.
    """
    item_id = get_item_id(item)
    user_content = build_user_content(item["context"])

    chosen_block = item["chosen"]
    rejected_block = item["rejected"]

    chosen_preds = wrap_predictions(extract_predictions(chosen_block), item_id)
    rejected_preds = wrap_predictions(extract_predictions(rejected_block), item_id)

    chosen_output = format_assistant_output(chosen_block, chosen_preds)
    rejected_output = format_assistant_output(rejected_block, rejected_preds)

    return {
        "conversations": [{"from": "human", "value": user_content}],
        "chosen": {"from": "gpt", "value": chosen_output},
        "rejected": {"from": "gpt", "value": rejected_output},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process synthesized_raw JSONL into SFT or preference JSON."
    )
    parser.add_argument("--input", type=str, required=False, default='data\synthesized_raw\LMSYS.jsonl', help="Path to synthesized_raw JSONL.")
    parser.add_argument("--output", type=str, required=False, default='data\processed\LMSYS_pref.json', help="Path to output JSON.")
    parser.add_argument("--mode", type=str, choices=["sft", "pref"], default="pref")
    args = parser.parse_args()

    data = read_jsonl(args.input)

    if args.mode == "sft":
        output_list = [convert_to_sft(item) for item in data]
    else:
        output_list = [convert_to_pref(item) for item in data]

    write_json(args.output, output_list)
    print(f"[Done] {len(output_list)} samples written to: {args.output}")


if __name__ == "__main__":
    main()
