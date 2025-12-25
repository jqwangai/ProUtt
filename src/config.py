from dataclasses import dataclass


@dataclass(frozen=True)
class SynthesisConfig:
    # Model and API
    base_url: str
    api_key_env: str
    model_id: str

    # Thresholds and limits
    high_confidence_threshold: float = 0.8
    low_confidence_threshold: float = 0.3
    max_pred_nums: int = 4

    # Concurrency
    max_concurrency: int = 10

    # I/O
    input_file: str = r"data\raw\LMSYS.jsonl"
    output_file: str = r"data\synthesized_raw\LMSYS.jsonl"
