"""Single source of truth for residual-flow state and conditioning channels."""

from __future__ import annotations

from dataclasses import asdict, dataclass


STATE_CHANNEL_NAMES = ("sdf_left", "sdf_right")
LEGACY_CONDITIONING_CHANNEL_NAMES = (
    "cbct", "prob_left", "prob_right", "coarse_sdf_left", "coarse_sdf_right",
    "coord_x", "coord_y", "coord_z",
)
PROMPT3R_CONDITIONING_CHANNEL_NAMES = (
    "cbct", "prob_left", "prob_right", "coord_x", "coord_y", "coord_z",
)


class ConditioningContractError(ValueError):
    """Checkpoint, tensor, and requested channel contracts do not agree."""


@dataclass(frozen=True)
class ConditioningSpec:
    state_channels: int
    conditioning_channel_names: tuple[str, ...]
    conditioning_channels: int
    include_coarse_sdf: bool
    contract_version: str

    def to_dict(self):
        payload = asdict(self)
        payload["conditioning_channel_names"] = list(self.conditioning_channel_names)
        return payload


def resolve_conditioning_spec(cfg=None):
    cfg = cfg or {}
    include = bool(cfg.get("cond_include_coarse_sdf", True))
    names = (LEGACY_CONDITIONING_CHANNEL_NAMES if include
             else PROMPT3R_CONDITIONING_CHANNEL_NAMES)
    version = "iac_flow_cond_v1_legacy8" if include else "iac_flow_cond_v2_prompt3r6"
    return ConditioningSpec(
        state_channels=len(STATE_CHANNEL_NAMES),
        conditioning_channel_names=names,
        conditioning_channels=len(names),
        include_coarse_sdf=include,
        contract_version=version,
    )


def validate_conditioning_tensor(tensor, spec, *, channel_axis=1, context="conditioning"):
    actual = int(tensor.shape[channel_axis])
    if actual != spec.conditioning_channels:
        raise ConditioningContractError(
            f"{context} channel mismatch: expected {spec.conditioning_channels} "
            f"{list(spec.conditioning_channel_names)}, got {actual}")
    return tensor


def validate_checkpoint_contract(checkpoint, spec, *, legacy_compatibility=False):
    metadata = checkpoint.get("channel_contract")
    if metadata is None:
        if not legacy_compatibility:
            raise ConditioningContractError(
                "checkpoint is missing channel_contract metadata; set "
                "legacy_compatibility=true only for an audited legacy checkpoint")
        if not spec.include_coarse_sdf or spec.conditioning_channels != 8:
            raise ConditioningContractError(
                "metadata-free legacy checkpoints are accepted only with the legacy 8-channel contract")
        return spec
    expected = spec.to_dict()
    normalized = dict(metadata)
    normalized["conditioning_channel_names"] = list(
        normalized.get("conditioning_channel_names", ()))
    if normalized != expected:
        raise ConditioningContractError(
            f"checkpoint channel contract mismatch: expected {expected}, got {normalized}")
    return spec


LEGACY_SPEC = resolve_conditioning_spec({"cond_include_coarse_sdf": True})
FLOW_STATE_CH = LEGACY_SPEC.state_channels
COND_CH = LEGACY_SPEC.conditioning_channels
