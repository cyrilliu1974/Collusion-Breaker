"""
aim_adapter.py
==============
橋接器：AIM Repo (步驟1-3) ↔ Collusion Framework (步驟4-6)

這個檔案是唯一知道兩邊格式的地方。
兩邊的程式都不需要知道對方的存在，只需要 import 這個模組。

使用方式：
    from aim_adapter import AIMAdapter
"""

import numpy as np


class AIMAdapter:
    """
    把 multi_agent_game() 的輸出轉換成 framework 標準格式，
    或反向把 framework 的參數轉換成 multi_agent_game() 的輸入。
    """

    def __init__(self, K: int = 32):
        """
        K : Codebook 大小，對應 aim_main.py 的 --K 參數
        """
        self.K = K

    # ──────────────────────────────────────────────────────────────────
    # 方向 A：aim_main.py → framework
    # ──────────────────────────────────────────────────────────────────
    def to_framework(self,
                     joint_hist:          list,
                     obs_acc_hist:        list,
                     shuffle_hist:        list,
                     encoding_inds_hist:  list,
                     penalty_hist:        list = None) -> dict:
        """
        輸入（來自 multi_agent_game() 的回傳值）：
            joint_hist          : joint_rewards_history
            obs_acc_hist        : obs_accuracies_history
            shuffle_hist        : shuffle_rounds_history
            encoding_inds_hist  : encoding_inds_history（需在 aim_main.py 收集）
            penalty_hist        : penalty_rounds_history（可選）

        輸出（framework 標準格式）：
            dict，可直接傳入 validate_thresholds() 或 perform_threshold_sweep()
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
    # 方向 B：framework 參數 → aim_main.py 輸入
    # ──────────────────────────────────────────────────────────────────
    def to_aim_kwargs(self,
                      threshold_penalty: float,
                      threshold_shuffle: float,
                      rounds:            int,
                      **kwargs) -> dict:
        """
        把 framework 的 sweep 參數轉成 multi_agent_game() 的 kwargs。

        使用方式：
            kwargs = adapter.to_aim_kwargs(tp, ts, rounds=1500)
            multi_agent_game(vqvae, aim_dict, **kwargs)
        """
        return {
            "rounds":                 rounds,
            "enable_reward_shaping":  True,
            "enable_codebook_shuffle": True,
            "threshold_penalty":      threshold_penalty,
            "threshold_shuffle":      threshold_shuffle,
            **kwargs,   # 其他額外參數透傳
        }

    # ──────────────────────────────────────────────────────────────────
    # 內部工具
    # ──────────────────────────────────────────────────────────────────
    def _build_codebook_usages(self, encoding_inds_hist: list) -> list:
        """
        把每回合的單一 encoding index 轉成 K 維 count 向量。

        encoding_inds_hist 可以是：
            - List[int]           : 每回合一個 index
            - List[List[int]]     : 每回合多個 index（aim_seq_len > 1）
        """
        usages = []
        for entry in encoding_inds_hist:
            indices = [entry] if isinstance(entry, int) else list(entry)
            counts = np.bincount(indices, minlength=self.K).astype(float)
            usages.append(counts)
        return usages