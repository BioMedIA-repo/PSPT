"""Strict PSPT checkpoint loader."""

from pathlib import Path

import torch

from .clam import CLAMMBInferenceHead
from .inference_model import PSPTInferenceModel


SCHEMA_VERSION = "pspt-inference-v1"


class PSPTCheckpointError(RuntimeError):
    pass


def _require(mapping, key, context):
    if key not in mapping:
        raise PSPTCheckpointError(
            f"Missing {context}.{key} in PSPT checkpoint."
        )
    return mapping[key]


def load_pspt_model(
    checkpoint_path,
    conch_backbone_checkpoint=None,
    device="cuda:0",
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        if "hyper_parameters" in checkpoint:
            raise PSPTCheckpointError(
                "Legacy pre-PSPT checkpoint schema detected. "
                "This public inference release requires a "
                f"{SCHEMA_VERSION} checkpoint."
            )
        raise PSPTCheckpointError(
            f"Unsupported checkpoint schema: "
            f"{checkpoint.get('schema_version')!r}."
        )

    config = _require(checkpoint, "config", "checkpoint")
    encoder_name = _require(config, "encoder", "config").upper()
    model_config = _require(config, "model", "config")
    scpm_config = _require(config, "scpm", "config")
    fprd_config = _require(config, "fprd", "config")
    inference_config = _require(config, "inference", "config")

    common = {
        "num_prompt_tokens": model_config["num_prompt_tokens"],
        "prompt_dropout": model_config["prompt_dropout"],
        "scpm_latent_dim": scpm_config["latent_dim"],
        "fprd_enabled": fprd_config["enabled"],
        "fprd_neighbors": fprd_config["neighbors"],
        "fprd_temperature": fprd_config["temperature"],
        "fprd_time_max": fprd_config["time_max"],
        "fprd_iterations": fprd_config["iterations"],
    }
    if encoder_name == "UNI":
        from .uni import UNIPSPTBackbone

        backbone = UNIPSPTBackbone(**common)
    elif encoder_name == "CONCH":
        if conch_backbone_checkpoint is None:
            raise PSPTCheckpointError(
                "--conch-backbone-checkpoint is required for CONCH."
            )
        from .conch import CONCHPSPTBackbone

        backbone = CONCHPSPTBackbone(
            backbone_checkpoint=conch_backbone_checkpoint,
            image_size=model_config["image_size"],
            **common,
        )
    else:
        raise PSPTCheckpointError(
            f"Unsupported encoder {encoder_name!r}; use UNI or CONCH."
        )

    task = config.get("task", {"type": "classification"})
    if task["type"] != "classification":
        raise PSPTCheckpointError(
            f"Unsupported task type {task['type']!r}."
        )
    classifier = CLAMMBInferenceHead(
        size=model_config["clam_size"],
        class_count=model_config["class_count"],
    )
    model = PSPTInferenceModel(
        backbone=backbone,
        classifier=classifier,
        evaluation_chunk_size=inference_config[
            "evaluation_chunk_size"
        ],
        normalization_mean=inference_config[
            "normalization_mean"
        ],
        normalization_std=inference_config[
            "normalization_std"
        ],
    )
    state_dict = _require(checkpoint, "state_dict", "checkpoint")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise PSPTCheckpointError(
            "Checkpoint tensors do not match the PSPT public model."
        ) from error
    return model.to(device).eval(), config
