"""
aim_adapter.py
==============
Adapter: AIM Repo (Steps 1-3) <-> Collusion Framework (Steps 4-6)

This file is the only place that knows the format of both sides.
The two programs do not need to know about each other's existence, they just need to import this module.

Usage:
    from aim_adapter import AIMAdapter
"""

import numpy as np


class AIMAdapter:
    """
    Converts the output of multi_agent_game() into the standard framework format,
    or conversely, converts the framework's parameters into input for multi_agent_game().
    """

    def __init__(self, K: int = 32):
        """
        K : Codebook size, corresponds to the --K parameter in aim_main.py
        """
        self.K = K

    # ──────────────────────────────────────────────────────────────────
    # Direction A: aim_main.py -> framework
    # ──────────────────────────────────────────────────────────────────
    def to_framework(self,
                     joint_hist:          list,
                     obs_acc_hist:        list,
                     shuffle_hist:        list,
                     encoding_inds_hist:  list,
                     penalty_hist:        list = None) -> dict:
        """
        Input (returns from multi_agent_game()):
            joint_hist          : joint_rewards_history
            obs_acc_hist        : obs_accuracies_history
            shuffle_hist        : shuffle_rounds_history
            encoding_inds_hist  : encoding_inds_history (needs to be collected in aim_main.py)
            penalty_hist        : penalty_rounds_history (optional)

        Output (standard framework format):
            dict, can be directly passed to validate_thresholds() or perform_threshold_sweep()
        """
        codebook_usages = self._build_codebook_usages(encoding_inds_hist)

        return {
            "joint_rewards":         np.array(joint_hist,   dtype=float),
            "observer_accs":   np.array(obs_acc_hist, dtype=float),
            "codebook_usages": codebook_usages,
            "shuffle_rounds":  shuffle_hist,
            "penalty_rounds":  penalty_hist or [],
        }

    # ──────────────────────────────────────────────────────────────────
    # Direction B: framework parameters -> aim_main.py input
    # ──────────────────────────────────────────────────────────────────
    def to_aim_kwargs(self,
                      threshold_penalty: float,
                      threshold_shuffle: float,
                      rounds:            int,
                      **kwargs) -> dict:
        """
        Converts framework sweep parameters into kwargs for multi_agent_game().

        Usage:
            kwargs = adapter.to_aim_kwargs(tp, ts, rounds=1500)
            multi_agent_game(vqvae, aim_dict, **kwargs)
        """
        return {
            "rounds":                 rounds,
            "enable_reward_shaping":  True,
            "enable_codebook_shuffle": True,
            "threshold_penalty":      threshold_penalty,
            "threshold_shuffle":      threshold_shuffle,
            **kwargs,   # Pass through other additional parameters
        }

    # ──────────────────────────────────────────────────────────────────
    # Internal tools
    # ──────────────────────────────────────────────────────────────────
    def _build_codebook_usages(self, encoding_inds_hist: list) -> list:
        """
        Converts a single encoding index from each round into a K-dimensional count vector.

        encoding_inds_hist can be:
            - List[int]           : One index per round
            - List[List[int]]     : Multiple indices per round (aim_seq_len > 1)
        """
        usages = []
        for entry in encoding_inds_hist:
            indices = [entry] if isinstance(entry, int) else list(entry)
            counts = np.bincount(indices, minlength=self.K).astype(float)
            usages.append(counts)
        return usages