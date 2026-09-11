from .components import score_from_components
from .config import SequenceScoreConfig
from .features import ObservationFeatures, UpdateFeatures
from .overlap import prefix_overlap_counts

import torch


def _excluded_token_ids(update: UpdateFeatures, observation: ObservationFeatures) -> set[int]:
    del update, observation
    return set()


def score_pair(
    update_features: UpdateFeatures,
    observation_features: ObservationFeatures,
    config: SequenceScoreConfig,
) -> dict[str, object]:
    if update_features.hidden.dim() != 3:
        raise ValueError("Update hidden must have shape [T, L, d].")
    if observation_features.hidden.dim() != 2:
        raise ValueError("Observation hidden must have shape [L, d].")

    excluded = _excluded_token_ids(update_features, observation_features) if config.exclude_special_tokens_from_overlap else set()
    kembd = prefix_overlap_counts(
        update_input_ids=update_features.input_ids,
        label_positions=update_features.label_positions,
        observation_prompt_ids=observation_features.prompt_ids,
        mode=config.kembd_mode,
        excluded_token_ids=excluded,
    )

    hh_all = torch.einsum("tld,ld->tl", update_features.hidden.float(), observation_features.hidden.float())

    uniform_gg = update_features.gradients.float() @ observation_features.option_gradients.float().T
    uniform_gwwg = (
        update_features.projected_gradients.float() @ observation_features.projected_option_gradients.float().T
    )
    uniform = score_from_components(
        gg=uniform_gg,
        hh_all=hh_all,
        gwwg=uniform_gwwg,
        kembd=kembd,
        objective="uniform_options",
    )

    ifmass_gg = update_features.gradients.float() @ observation_features.ifmass_gradient.float().T
    ifmass_gwwg = update_features.projected_gradients.float() @ observation_features.projected_ifmass_gradient.float().T
    ifmass = score_from_components(
        gg=ifmass_gg,
        hh_all=hh_all,
        gwwg=ifmass_gwwg,
        kembd=kembd,
        objective="ifmass",
    )

    return {
        "uniform_options": uniform,
        "ifmass": ifmass,
        "diagnostics": {
            "gg_shape": list(uniform_gg.shape),
            "gwwg_shape": list(uniform_gwwg.shape),
            "hh_all_shape": list(hh_all.shape),
            "kembd_shape": list(kembd.shape),
        },
    }
