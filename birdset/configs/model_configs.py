from dataclasses import dataclass


@dataclass
class PretrainInfoConfig:
    """
    hf_path: HF datasets repo path
    hf_name: subset name of current training
    hf_pretrain_name: HF subset name of used for pretraining, assumed to be from hf_path repo
    valid_test_only: if masking should only be applied in valid/test/predict stage of trainer
    """

    hf_path: str = "DBD-research-group/BirdSet"
    hf_name: str = "HSN"
    hf_pretrain_name: str | None = None
    valid_test_only: bool = False
