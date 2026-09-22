"""Saved parameter presets for the 3D Schelling model."""
import random
import numpy as np

DEFAULT_PRESET = {
    "delay_per_plot": 0.003,
    "similarity_threshold": 0.2,
    "segregation_threshold": 0.75,
    "consensus_threshold": 0.2,
    "consensus_weight": 0.06,
    "floodfill_threshold": 0.01,
    "opinion_group_threshold": 0.02,
    "seed": 2,
    "graphs": True,
    "consensus": True,
    "segregation": True,
    "grouping_graphs": True,
}

PRESETS = {
    "default": {
        "delay_per_plot": 0.003,
        "similarity_threshold": 0.2,
        "segregation_threshold": 0.75,
        "consensus_threshold": 0.2,
        "consensus_weight": 0.06,
        "floodfill_threshold": 0.01,
        "opinion_group_threshold": 0.02,
        "seed": 2,
        "graphs": True,
        "consensus": True,
        "segregation": True,
        "grouping_graphs": True,
    },
    # get_preset() randomises similarity_threshold, segregation_threshold,
    # and consensus_threshold independently in [0, 1); other values stay default.
    "Randomise": DEFAULT_PRESET.copy(),
    "clusters_formed": {
        "delay_per_plot": 0.003,
        "similarity_threshold": 0.2,
        "segregation_threshold": 0.75,
        "consensus_threshold": 0.2,
        "consensus_weight": 0.06,
        "floodfill_threshold": 0.2,
        "opinion_group_threshold": 0.02,
        "seed": 2,
        "graphs": True,
        "consensus": True,
        "segregation": True,
        "grouping_graphs": True,
    },
    "mean_similarity_decreases": {
        "delay_per_plot": 0.003,
        "similarity_threshold": 0.2,
        "segregation_threshold": 0.75,
        "consensus_threshold": 0.1,
        "consensus_weight": 0.06,
        "floodfill_threshold": 0.2,
        "opinion_group_threshold": 0.02,
        "seed": 2,
        "graphs": True,
        "consensus": True,
        "segregation": True,
        "grouping_graphs": True,
    },
    "mean_similarity_stays_same": {
        "delay_per_plot": 0.003,
        "similarity_threshold": 0.2,
        "segregation_threshold": 0.75,
        "consensus_threshold": 0.15,
        "consensus_weight": 0.06,
        "floodfill_threshold": 0.2,
        "opinion_group_threshold": 0.02,
        "seed": 2,
        "graphs": True,
        "consensus": True,
        "segregation": True,
        "grouping_graphs": True,
    },
    "snake": {
        "delay_per_plot": 0.0001,
        "similarity_threshold": 0.2,
        "segregation_threshold": 0.8,
        "consensus_threshold": 0.16,
        "consensus_weight": 0.06,
        "floodfill_threshold": 0.2,
        "opinion_group_threshold": 0.02,
        "seed": 2,
        "graphs": True,
        "consensus": True,
        "segregation": True,
        "grouping_graphs": True,
    },
    "yin_and_yang": {
        "delay_per_plot": 0.003,
        "similarity_threshold": 0.2,
        "segregation_threshold": 0.75,
        "consensus_threshold": 0.17,
        "consensus_weight": 0.06,
        "floodfill_threshold": 0.2,
        "opinion_group_threshold": 0.02,
        "seed": 2,
        "graphs": True,
        "consensus": True,
        "segregation": True,
        "grouping_graphs": True,
    },
}


def get_preset(name):
    """Return a copy of a preset so callers cannot change the saved values."""
    if name not in PRESETS:
        available = ", ".join(PRESETS)
        raise ValueError(f"Unknown preset {name!r}. Choose from: {available}")
    preset = PRESETS[name].copy()
    if name == "Randomise":
        for parameter in (
            "similarity_threshold",
            "segregation_threshold",
            "consensus_threshold",
        ):
            preset[parameter] = random.random()

    return preset
