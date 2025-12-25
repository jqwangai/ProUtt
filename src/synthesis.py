import asyncio
import random
from typing import Any, Dict, List

from tqdm.asyncio import tqdm_asyncio

from .config import SynthesisConfig
from .client import build_client, chat_once_json
from .utils import messages2history_round, accumulate_token_usage, top_similarity
from .io_utils import read_jsonl, append_jsonl, sum_usage_from_jsonl
from .prompts import *


async def negative_construct_v2(
    client,
    cfg: SynthesisConfig,
    intent_tree: Dict[str, Any],
    gt_insight: str,
    gt_path: Any,
    negative_label: List[str],
    chat_history: str,
    utterance_category_reasons: List[Dict[str, Any]],
    insight_reason_original: Dict[str, Any],
    usage_all: Dict[str, int],
) -> Dict[str, Any]:
    """
    Construct a rejected sample by creating an incorrect path and revising reasoning.
    """
    if len(negative_label) != 0:
        incorrect_view_text, usage_item = await chat_once_json(
            client=client,
            model_id=cfg.model_id,
            system_prompt=Incorrect_Path_With_Reference_Sys_Prompt,
            user_content=Incorrect_Path_With_Reference_User_Prompt.format(
                intent_tree=intent_tree,
                gt_insight=gt_insight,
                gt_path=gt_path,
                error_user_input=negative_label,
            ),
        )
        accumulate_token_usage(usage_all, usage_item)
        incorrect_path = incorrect_view_text["path"]
    else:
        incorrect_view_text, usage_item = await chat_once_json(
            client=client,
            model_id=cfg.model_id,
            system_prompt=Incorrect_Path_Sys_Prompt,
            user_content=Incorrect_Path_User_Prompt.format(
                intent_tree=intent_tree,
                gt_insight=gt_insight,
                gt_path=gt_path,
            ),
        )
        accumulate_token_usage(usage_all, usage_item)
        incorrect_path = incorrect_view_text["path"]

    utterance_category_reason = random.choice(utterance_category_reasons)
    category = utterance_category_reason["category"]

    if gt_insight == "Mining":
        revised_insight_reason, usage_item = await chat_once_json(
            client=client,
            model_id=cfg.model_id,
            system_prompt=Mining_Revise_Sys_Prompt,
            user_content=Mining_Revise_User_Prompt.format(
                context=chat_history,
                intent_tree=intent_tree,
                category=category,
                predict_reasoning=insight_reason_original["mining_view"]["reasoning"],
                path=incorrect_path,
            ),
        )
        accumulate_token_usage(usage_all, usage_item)
    else:
        revised_insight_reason, usage_item = await chat_once_json(
            client=client,
            model_id=cfg.model_id,
            system_prompt=Explore_Revise_Sys_Prompt,
            user_content=Explore_Revise_User_Prompt.format(
                context=chat_history,
                intent_tree=intent_tree,
                category=category,
                predict_reasoning=insight_reason_original["explore_view"]["reasoning"],
                path=incorrect_path,
            ),
        )
        accumulate_token_usage(usage_all, usage_item)

    # Normalize wording in revised reasoning
    revised_insight_reason["revised"] = revised_insight_reason["revised"].replace("mining path", "mining view")
    revised_insight_reason["revised"] = revised_insight_reason["revised"].replace("exploration path", "exploration view")

    # Only keep half predictions for the revised side
    revised_insight_reason["predictions"] = revised_insight_reason["predictions"][: cfg.max_pred_nums // 2]

    rejected: Dict[str, Any] = {
        "incorrect_path": incorrect_path,
        "intent_tree": intent_tree,
        "utterance_category_reason": utterance_category_reason,
        "insight_reason": {},
    }

    if gt_insight == "Mining":
        rejected["insight_reason"]["mining_view"] = {
            "reasoning": revised_insight_reason["revised"],
            "predictions": revised_insight_reason["predictions"],
        }
        rejected["insight_reason"]["explore_view"] = insight_reason_original["explore_view"]
    else:
        rejected["insight_reason"]["mining_view"] = insight_reason_original["mining_view"]
        rejected["insight_reason"]["explore_view"] = {
            "reasoning": revised_insight_reason["revised"],
            "predictions": revised_insight_reason["predictions"],
        }

    return rejected


async def process_item(client, cfg: SynthesisConfig, item: Dict[str, Any]) -> None:
    """
    Process one training item and append the synthesized record into output jsonl.
    """
    usage_all = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    chat_history = messages2history_round(item["context"])

    new_json = dict(item)
    new_json["chat_history"] = chat_history

    # 1) Intent tree construction
    intent_tree, usage_item = await chat_once_json(
        client=client,
        model_id=cfg.model_id,
        system_prompt=INITIAL_EXTRACTION_SYS_PROMPT,
        user_content=INITIAL_EXTRACTION_USER_PROMPT.format(chat_history=chat_history),
    )
    accumulate_token_usage(usage_all, usage_item)
    new_json["intent_tree"] = intent_tree

    # 2) Utterance category reasoning
    utterance_category_reason, usage_item = await chat_once_json(
        client=client,
        model_id=cfg.model_id,
        system_prompt=Utterance_Classification_Sys_Prompt,
        user_content=Utterance_Classification_User_Prompt.format(
            chat_history=chat_history, ground_truth=item["label"]
        ),
    )
    accumulate_token_usage(usage_all, usage_item)
    new_json["utterance_category_reason"] = utterance_category_reason

    # 3) Utterance category ground truth (derived from last assistant + user label)
    utterance_category, usage_item = await chat_once_json(
        client=client,
        model_id=cfg.model_id,
        system_prompt=Utterance_Classification_GT_Sys_Prompt,
        user_content=Utterance_Classification_GT_User_Prompt.format(
            assistant_message=item["context"][-1],
            user_message=item["label"],
        ),
    )
    accumulate_token_usage(usage_all, usage_item)
    category = utterance_category["predicted_category"]
    new_json["utterance_category_gt"] = utterance_category

    # 4) Insight reasoning (mining view + exploration view)
    insight_reason, usage_item = await chat_once_json(
        client=client,
        model_id=cfg.model_id,
        system_prompt=Insight_Sys_Prompt,
        user_content=Insight_User_Prompt.format(
            chat_history=chat_history,
            intent_tree=intent_tree,
            category=category,
        ),
    )
    accumulate_token_usage(usage_all, usage_item)
    new_json["insight_reason"] = insight_reason

    # 5) Evaluate predictions
    all_predictions = insight_reason["mining_view"]["predictions"] + insight_reason["explore_view"]["predictions"]
    if len(all_predictions) != cfg.max_pred_nums:
        print(f"item id:{item['sample_id']}: predictions number error")

    predictions_text = ""
    for i, v in enumerate(all_predictions):
        predictions_text += f"Predictive Input {i + 1}: {v}\n"

    evaluate_reason, usage_item = await chat_once_json(
        client=client,
        model_id=cfg.model_id,
        system_prompt=Evaluate_Sys_Prompt,
        user_content=Evaluate_User_Prompt.format(
            context=chat_history,
            label=item["label"],
            predict_input=predictions_text,
        ),
    )
    accumulate_token_usage(usage_all, usage_item)
    new_json["evaluate_reason"] = evaluate_reason

    top_sim = top_similarity(evaluate_reason)
    new_json["top_sim"] = top_sim

    # 6) Get GT insight and GT path
    gt_insight_path, usage_item = await chat_once_json(
        client=client,
        model_id=cfg.model_id,
        system_prompt=GT_Insight_Path_Sys_Prompt,
        user_content=GT_Insight_Path_User_Prompt.format(
            context=chat_history,
            intent_tree=intent_tree,
            label=item["label"],
        ),
    )
    accumulate_token_usage(usage_all, usage_item)
    new_json["gt_insight_path"] = gt_insight_path

    insight_gt = gt_insight_path["insight"]
    revised_path = gt_insight_path["path"]

    # 7) Build chosen and rejected
    chosen: Dict[str, Any] = {}
    rejected: Dict[str, Any] = {}

    if top_sim >= cfg.high_confidence_threshold:
        chosen = {
            "intent_tree": intent_tree,
            "utterance_category_reason": [x for x in utterance_category_reason if x["category"] == category][0],
            "insight_reason": insight_reason,
        }
        rejected = await negative_construct_v2(
            client, cfg, intent_tree, insight_gt, revised_path[0],
            item["negative_label"], chat_history, utterance_category_reason,
            insight_reason, usage_all
        )

    elif top_sim >= cfg.low_confidence_threshold:
        gt_path = revised_path[0]
        random.shuffle(revised_path)

        if insight_gt == "Mining":
            revised_insight_reason, usage_item = await chat_once_json(
                client=client,
                model_id=cfg.model_id,
                system_prompt=Mining_Revise_Sys_Prompt,
                user_content=Mining_Revise_User_Prompt.format(
                    context=chat_history,
                    intent_tree=intent_tree,
                    category=category,
                    predict_reasoning=insight_reason["mining_view"]["reasoning"],
                    path=revised_path,
                ),
            )
        else:
            revised_insight_reason, usage_item = await chat_once_json(
                client=client,
                model_id=cfg.model_id,
                system_prompt=Explore_Revise_Sys_Prompt,
                user_content=Explore_Revise_User_Prompt.format(
                    context=chat_history,
                    intent_tree=intent_tree,
                    category=category,
                    predict_reasoning=insight_reason["explore_view"]["reasoning"],
                    path=revised_path,
                ),
            )
        accumulate_token_usage(usage_all, usage_item)

        revised_insight_reason["revised"] = revised_insight_reason["revised"].replace("mining path", "mining view")
        revised_insight_reason["revised"] = revised_insight_reason["revised"].replace("exploration path", "exploration view")
        revised_insight_reason["predictions"] = revised_insight_reason["predictions"][: cfg.max_pred_nums // 2]
        new_json["revised_insight_reason"] = revised_insight_reason

        chosen = {
            "intent_tree": intent_tree,
            "utterance_category_reason": [x for x in utterance_category_reason if x["category"] == category][0],
            "insight_reason": {
                "mining_view": insight_reason["mining_view"],
                "explore_view": insight_reason["explore_view"],
            },
        }
        if insight_gt == "Mining":
            chosen["insight_reason"]["mining_view"] = {
                "reasoning": revised_insight_reason["revised"],
                "predictions": revised_insight_reason["predictions"],
            }
        else:
            chosen["insight_reason"]["explore_view"] = {
                "reasoning": revised_insight_reason["revised"],
                "predictions": revised_insight_reason["predictions"],
            }

        rejected = await negative_construct_v2(
            client, cfg, intent_tree, insight_gt, gt_path,
            item["negative_label"], chat_history, utterance_category_reason,
            insight_reason, usage_all
        )

    else:
        random.shuffle(revised_path)

        if insight_gt == "Mining":
            revised_insight_reason, usage_item = await chat_once_json(
                client=client,
                model_id=cfg.model_id,
                system_prompt=Mining_Revise_Sys_Prompt,
                user_content=Mining_Revise_User_Prompt.format(
                    context=chat_history,
                    intent_tree=intent_tree,
                    category=category,
                    predict_reasoning=insight_reason["mining_view"]["reasoning"],
                    path=revised_path,
                ),
            )
        else:
            revised_insight_reason, usage_item = await chat_once_json(
                client=client,
                model_id=cfg.model_id,
                system_prompt=Explore_Revise_Sys_Prompt,
                user_content=Explore_Revise_User_Prompt.format(
                    context=chat_history,
                    intent_tree=intent_tree,
                    category=category,
                    predict_reasoning=insight_reason["explore_view"]["reasoning"],
                    path=revised_path,
                ),
            )
        accumulate_token_usage(usage_all, usage_item)

        revised_insight_reason["revised"] = revised_insight_reason["revised"].replace("mining path", "mining view")
        revised_insight_reason["revised"] = revised_insight_reason["revised"].replace("exploration path", "exploration view")
        revised_insight_reason["predictions"] = revised_insight_reason["predictions"][: cfg.max_pred_nums // 2]
        new_json["revised_insight_reason"] = revised_insight_reason

        chosen = {
            "intent_tree": intent_tree,
            "utterance_category_reason": [x for x in utterance_category_reason if x["category"] == category][0],
            "insight_reason": {
                "mining_view": insight_reason["mining_view"],
                "explore_view": insight_reason["explore_view"],
            },
        }
        if insight_gt == "Mining":
            chosen["insight_reason"]["mining_view"] = {
                "reasoning": revised_insight_reason["revised"],
                "predictions": revised_insight_reason["predictions"],
            }
        else:
            chosen["insight_reason"]["explore_view"] = {
                "reasoning": revised_insight_reason["revised"],
                "predictions": revised_insight_reason["predictions"],
            }

        rejected = {
            "intent_tree": intent_tree,
            "utterance_category_reason": [x for x in utterance_category_reason if x["category"] == category][0],
            "insight_reason": insight_reason,
        }

    new_json["usage"] = usage_all
    new_json["chosen"] = chosen
    new_json["rejected"] = rejected

    append_jsonl(cfg.output_file, new_json)


async def process_sample(client, cfg: SynthesisConfig, sample: Dict[str, Any]) -> None:
    """
    Convert one LMSYS conversation sample into an item for synthesis.
    """
    conversation = sample["conversation"]
    rounds = len(conversation) // 2
    if rounds < 2:
        print(f"Skipping sample {sample['id']} because it has less than 2 rounds of messages")
        return

    random.seed(sample["id"])
    selected_round = random.randint(0, rounds - 2)

    item: Dict[str, Any] = {
        "sample_id": sample["id"],
        "item_id": 0,
        "context": conversation[:-2],
        "label": conversation[-2]["content"],
        "negative_label": [],
    }

    # Negative labels are taken from earlier turns, separated by at least one round.
    if len(conversation) - (selected_round * 2 + 6) >= 0:
        item["negative_label"].append("Incorrect Input 1: " + str(conversation[selected_round * 2 + 4]["content"]))
    if len(conversation) - (selected_round * 2 + 8) >= 0:
        item["negative_label"].append("Incorrect Input 2: " + str(conversation[selected_round * 2 + 6]["content"]))

    await process_item(client, cfg, item)


async def process_sample_with_semaphore(sample, semaphore, pbar, client, cfg):
    async with semaphore:
        try:
            await process_sample(client, cfg, sample)
        finally:
            pbar.update(1)


async def run(cfg: SynthesisConfig) -> None:
    client = build_client(cfg.base_url, cfg.api_key_env)
    samples = read_jsonl(cfg.input_file)

    semaphore = asyncio.Semaphore(cfg.max_concurrency)
    with tqdm_asyncio(total=len(samples), desc="Syn", ncols=100) as pbar:
        tasks = [
            process_sample_with_semaphore(sample, semaphore, pbar, client, cfg)
            for sample in samples
        ]
        await asyncio.gather(*tasks)

    usage_all = sum_usage_from_jsonl(cfg.output_file)
    print(f"Total usage: {usage_all}")


def main():
    # You can replace these defaults with argparse later if needed.
    cfg = SynthesisConfig(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key_env="ZAI_API_KEY",
        model_id="glm-4.6",
        input_file=r"data\raw\LMSYS.jsonl",
        output_file=r"data\synthesized_raw\LMSYS.jsonl",
        max_concurrency=10,
    )
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
