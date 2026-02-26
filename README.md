# AI Mother Tongue: Self-Emergent Communication in MARL via Endogenous Symbol Systems

This project is an implementation of the paper **"AI Mother Tongue: Self-Emergent Communication in MARL via Endogenous Symbol Systems"**, which introduces a novel framework for fostering self-emergent communication in multi-agent reinforcement learning (MARL). The full paper explores how agents can develop an **endogenous symbol system** using a Vector Quantized Variational Autoencoder (VQ-VAE), enabling them to compress observations into discrete symbols for internal reflection and communication, without relying on pre-defined protocols or artificial inductive biases.

> **New in this release**: This repository now includes a **Collusion Detection & Defense Framework** — a three-module extension (`vqvae_agents_AIM.py` + `aim_adapter.py` + `aim_collusion_framework.py`) that implements a scientifically rigorous 6-step methodology for detecting, quantifying, and verifying hidden collusion behavior in multi-agent systems. See the [Collusion Detection Framework](#collusion-detection-and-defense-framework) section for details.

-----

## Repository Structure

```
your_project/
├── vqvae_agents_AIM.py          # Core AIM training loop (Steps 1–3)
├── aim_adapter.py               # Data bridge between training and analysis (NEW)
├── aim_collusion_framework.py   # Collusion detection & verification (Steps 4–6) (NEW)
├── aim_dictionary_json.py       # AIM dictionary manager
├── enhanced_aim_dictionary.py   # Enhanced AIM dictionary with reflection records
├── analyze_aim.py               # AIM dictionary analysis tool
├── vqvae_agents_game_RIAL.py    # RIAT baseline (Positive Signalling + Listening)
└── vqvae_agents_game.py         # Original Intent Alignment baseline
```

-----

## `vqvae_agents_AIM.py` - Core Implementation of AI Mother Tongue Framework

`vqvae_agents_AIM.py` is the central script that embodies the **"AI Mother Tongue Framework."** It orchestrates the training of a Vector Quantized Variational Autoencoder (VQ-VAE) and its integration into a multi-agent cooperative game. This script demonstrates how agents can develop an **endogenous symbol system** for communication, moving beyond pre-defined communication protocols. The VQ-VAE enables agents to compress continuous environmental observations (e.g., MNIST images) into discrete, learned symbols (the "AI Mother Tongue"), which are then used for internal reflection and inter-agent communication.

### Key Functionality:

  * **VQ-VAE Training**: Trains a VQ-VAE on environmental observations to learn a discrete codebook, forming the basis of the symbolic language.
  * **Multi-Agent Game Simulation**: Runs a cooperative game where two agents (AgentA and AgentB) interact with the environment and each other.
  * **Emergent Communication**: Agents utilize the discrete symbols from the VQ-VAE codebook as their communication signals, with their meanings emerging through reinforcement learning and a **reflection mechanism**.
  * **Reflection Mechanism**: Implements strategies (e.g., `predictive_bias`) that encourage agents to align their internal states and actions with the emergent symbols, fostering meaningful communication and achieving "spontaneous semantic compression" and "Nash equilibrium-driven semantic convergence".
  * **Collusion Defense (Circuit Breaker)**: Implements a two-tier defense mechanism — Reward Shaping (penalty) and Codebook Shuffle — triggered by the Collusion Score when agents show signs of hidden coordination.
  * **AIM Dictionary Integration**: Records and manages the learned "Agent's Internal Monologue" (AIM) sequences and their associated contexts and human interpretations via `aim_dictionary.json`.
  * **Collusion Data Export**: Exports `encoding_inds_history` (VQ-VAE codebook usage per round) alongside reward and observer accuracy histories, enabling downstream analysis by `aim_collusion_framework.py` via `aim_adapter.py`.

### Command-Line Parameters for `vqvae_agents_AIM.py`

This script provides several configurable parameters via `argparse`, allowing users to control the VQ-VAE training, game dynamics, and the communication emergence process. These parameters can **override** the default values set within the script.

Users can specify the number of `--epochs` (default: 10, optimal: 10-20, range: 5-30) for VQ-VAE pre-training, determining how many times the VQ-VAE processes the entire dataset to learn its codebook. The total number of `--rounds` (default: 10000, optimal: 10000-20000, range: 5000-50000) for the multi-agent game simulation can also be set. The size of the VQ-VAE codebook, `--K` (default: 32, optimal: 32 or 64, range: 16, 32, 64, 128), directly determines the size of the discrete vocabulary available to the agents. The `--D` (default: 64, optimal: 64, range: 32, 64, 128) parameter controls the dimension of the VQ-VAE code vectors.

The `--aim_seq_len` (default: 2, optimal: 2, range: 1-3) defines how many discrete symbols constitute a single communicative "utterance" from an agent. The `--reflection_strategy` (default: `predictive_bias`, choices: `none`, `aim_context_value`, `predictive_bias`) selects the reflection mechanism for agents to learn from their internal symbols; `predictive_bias` is optimal. The impact of this mechanism is controlled by the `--reflection_coeff` (default: 0.05, optimal: 0.05 or 0.1, range: 0.01-0.2).

For reinforcement learning, `--gamma_rl` (default: 0.99, optimal: 0.99, range: 0.9-0.99) sets the discount factor for rewards, while `--entropy_coeff` (default: 0.01, optimal: 0.01 or 0.05, range: 0.001-0.1) provides an initial coefficient for entropy regularization to encourage exploration.

**Collusion Defense Parameters** (new):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable_reward_shaping` | False | Enable Defense A: penalize reward when Collusion Score exceeds `threshold_penalty` |
| `--enable_codebook_shuffle` | False | Enable Defense B: reset codebook weights when Collusion Score exceeds `threshold_shuffle` |
| `--threshold_penalty` | 12.0 | Collusion Score threshold that triggers reward penalty (-2.0 points) |
| `--threshold_shuffle` | 18.0 | Collusion Score threshold that triggers codebook shuffle (catastrophic reset) |

The **Collusion Score** is defined as:

```
Collusion Score = Joint Reward / (Observer Accuracy + 0.1)
```

A high score means agents are achieving high joint rewards while the external observer cannot predict their actions — a strong signal of hidden coordination through a private protocol.

### Example Execution with Optimal Parameters:

To run the "AI Mother Tongue" framework using `vqvae_agents_AIM.py` with a recommended set of parameters that have shown good results in fostering emergent communication:

```bash
python vqvae_agents_AIM.py --epochs 10 --rounds 10000 --K 32 --D 64 --aim_seq_len 2 --reflection_strategy predictive_bias --reflection_coeff 0.05 --gamma_rl 0.99 --entropy_coeff 0.05
```

To run with the collusion defense mechanisms enabled:

```bash
python vqvae_agents_AIM.py --epochs 10 --rounds 1500 --K 32 \
  --enable_reward_shaping --enable_codebook_shuffle \
  --threshold_penalty 12.0 --threshold_shuffle 18.0
```

-----

## Collusion Detection and Defense Framework

This section describes the three-module extension for **scientifically detecting, quantifying, and verifying hidden collusion behavior** in the AIM multi-agent system. It follows a rigorous 6-step research methodology:

| Step | Description | Implemented In |
|------|-------------|----------------|
| 1 | Design adversarial pressure field | `vqvae_agents_AIM.py` (BaselineObserver + Circuit Breaker) |
| 2 | Discover unexpected behavior | `vqvae_agents_AIM.py` (gray-zone oscillation, covert collusion) |
| 3 | Define the phenomena | Paper + Collusion Score formula |
| 4 | Formalize metrics | `aim_collusion_framework.py` |
| 5 | Derive boundary conditions | `aim_collusion_framework.py` |
| 6 | Verify with experiments | `aim_collusion_framework.py` |

Steps 1–3 are the **original research contributions** embedded in the training loop. Steps 4–6 are the **analytical and verification layer** provided by the framework. The two layers are connected by `aim_adapter.py`, which acts as a thin data bridge without coupling the two systems.

### Three Phases of Agent Behavior

The framework identifies and tracks four distinct behavioral phases:

| Phase | Color | Condition | Interpretation |
|-------|-------|-----------|----------------|
| **Collapse** | 🔴 Red | Reward variance high | Training diverged; codebook shuffle too aggressive |
| **Boundary Oscillation** | 🟠 Orange | Reward stable, JSD high | Agents testing the gray zone — partial hidden signals |
| **Covert Collusion** | 🟣 Purple | Reward high, Observer Acc low | Agents successfully hiding collusion; most dangerous state |
| **Transparent Convergence** | 🟢 Green | Reward high, Acc high, JSD low | Defense successful; agents forced into interpretable communication |

-----

## `aim_adapter.py` — Data Bridge (Steps 1–3 ↔ Steps 4–6)

`aim_adapter.py` is the **only module that knows the data format of both sides**. Neither `vqvae_agents_AIM.py` nor `aim_collusion_framework.py` needs to know about the other's existence. This design keeps the training logic and the analysis logic fully decoupled.

### What It Does

**Direction A — Training → Framework:**
Converts `multi_agent_game()` output into the standard format expected by the framework.

```python
from aim_adapter import AIMAdapter

adapter = AIMAdapter(K=32)   # K must match --K used in training

framework_data = adapter.to_framework(
    joint_hist,        # joint_rewards_history  from multi_agent_game()
    obs_acc_hist,      # obs_accuracies_history from multi_agent_game()
    shuffle_hist,      # shuffle_rounds_history from multi_agent_game()
    enc_inds_hist      # encoding_inds_history  from multi_agent_game() [NEW]
)
# framework_data is a dict ready for aim_collusion_framework.py
```

**Direction B — Framework → Training:**
Converts framework sweep parameters into `multi_agent_game()` keyword arguments.

```python
kwargs = adapter.to_aim_kwargs(
    threshold_penalty=12.0,
    threshold_shuffle=18.0,
    rounds=1500
)
multi_agent_game(vqvae, aim_dict, **kwargs)
```

### Why `encoding_inds_history` Is Needed

The framework's `jsd_rep_shift()` function measures **semantic distribution shift** — how much the agent's symbol usage pattern changes over time. This requires knowing which codebook slot was used each round.

The VQ-VAE already computes this internally (`encoding_inds` from `VectorQuantizer.forward()`), but the original code discarded it. The updated `vqvae_agents_AIM.py` now collects it with a single `with torch.no_grad()` call per round, completely without affecting training gradients.

```
Raw output:   [7, 7, 2, 7, 3, 7, ...]      ← one integer per round
After adapter: [[0,0,1,0,0,0,0,1,...],      ← K-dimensional count vector
                [0,0,0,0,0,0,0,1,...], ...]
Framework uses: JSD between consecutive      ← semantic stability metric
                window distributions
```

-----

## `aim_collusion_framework.py` — Analysis & Verification (Steps 4–6)

`aim_collusion_framework.py` is a **standalone analytical framework** that can be used with any multi-agent system that produces the three core signals: `joint_reward`, `observer_accuracy`, and `codebook_usage_distribution`. It is not specific to the AIM architecture.

### Core Metrics (Step 4)

| Metric | Function | What It Measures |
|--------|----------|-----------------|
| **Surface Stability** | `sliding_variance()` | Rolling variance of joint reward over a 50-round window |
| **Semantic Stability** | `jsd_rep_shift()` | Jensen-Shannon Divergence between consecutive codebook usage distributions |
| **L2 Rep Shift** | `norm_rep_shift()` | L2 norm of codebook usage vector change over time (complementary to JSD) |
| **Collusion Score** | `collusion_scores()` | Joint Reward / (Observer Accuracy + ε) per round |

**Why JSD instead of `np.var()` on codebook indices:**
Codebook indices are categorical (discrete labels), not ordinal numbers. The difference between index 2 and index 30 carries no magnitude information. JSD correctly treats them as probability distributions, measuring how much the usage pattern has shifted — which is the actual signal for semantic stability.

### Phase Boundary Derivation (Step 5)

```bash
python aim_collusion_framework.py --mode sweep
```

This performs a grid search over `(threshold_penalty, threshold_shuffle)` parameter space using `np.linspace` with 10 points per axis. For each combination, it runs `num_seeds` experiments, classifies the dominant behavioral phase, and outputs:

- `sweep_data.csv` — raw data for every parameter combination
- `figures/phase_diagram.png` — two-panel heatmap (Phase classification + Mean Reward)

The `auto_boundary()` function automatically derives classification thresholds from the early training dynamics (first 50 rounds), eliminating manual tuning.

### Experimental Verification (Step 6)

```bash
python aim_collusion_framework.py --mode validate
```

For specific threshold configurations (default: `(3.0, 5.0)`, `(9.0, 14.0)`, `(12.0, 18.0)`), this runs `num_seeds` full experiments and produces:

- **Four-panel validation plots** per configuration (Reward ± STD, Observer Accuracy, JSD Rep Shift, Collusion Score)
- **Pseudo-stable round count**: Number of rounds where reward is stable but semantics are still drifting (the "gray zone")
- **Welch's t-test**: Statistical significance of differences between configurations (reported as t-statistic and p-value)

### Command-Line Parameters for `aim_collusion_framework.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `full` | `sweep` = phase diagram only; `validate` = deep validation only; `full` = both |
| `--seeds` | `5` | Number of random seeds per configuration (more = more reliable statistics) |
| `--rounds` | `1500` | Rounds per experiment |

-----

## How to Run: Three Scenarios

### Scenario 1 — Run AIM Training Only (Steps 1–3)

This is identical to the original usage. No framework required.

```bash
python vqvae_agents_AIM.py \
  --rounds 1500 \
  --enable_reward_shaping \
  --enable_codebook_shuffle \
  --threshold_penalty 12.0 \
  --threshold_shuffle 18.0
```

**Output:** `marl_collusion_evolution.png` — dual-axis plot showing Joint Reward (red) and Observer Accuracy (blue) over training rounds, with orange star markers at each Codebook Shuffle event.

**When to use:** When you want to quickly observe training dynamics for a specific threshold configuration, or when iterating on the training architecture itself.

---

### Scenario 2 — Run Full Collusion Analysis (Steps 4–6)

This calls the framework directly. The framework internally calls `vqvae_agents_AIM.py` via the adapter for each parameter combination.

```bash
# Step 5 only: scan threshold space → phase diagram
python aim_collusion_framework.py --mode sweep --seeds 3

# Step 6 only: deep validate specific thresholds
python aim_collusion_framework.py --mode validate --seeds 5

# Both steps (recommended for paper-quality results)
python aim_collusion_framework.py --mode full --seeds 5 --rounds 1500
```

**Output:**
```
figures/
├── phase_diagram.png                  # Phase heatmap across threshold space
├── validate_tp=3.0_ts=5.0.png        # Collapse regime: 4-panel validation
├── validate_tp=9.0_ts=14.0.png       # Boundary regime: 4-panel validation
└── validate_tp=12.0_ts=18.0.png      # Transparent regime: 4-panel validation
sweep_data.csv                         # Raw sweep data (all threshold combinations)
validation_results.json                # Final metrics + Welch t-test results
```

**When to use:** When generating figures and statistical results for the paper's Experiments section.

---

### Scenario 3 — Single Training Run with Immediate Analysis

Run one training session and immediately inspect steps 4–6 metrics in the same script. Add the following to the end of `vqvae_agents_AIM.py`'s `main` block (after `visualize()`):

```python
from aim_collusion_framework import (
    sliding_variance, jsd_rep_shift,
    classify_phase, align_metrics,
    WINDOW_SIZE
)

rv  = sliding_variance(framework_data["rewards"], WINDOW_SIZE)
jsd = jsd_rep_shift(framework_data["codebook_usages"], WINDOW_SIZE)
rv, jsd = align_metrics(rv, jsd)
phase = classify_phase(rv, jsd, framework_data["observer_accs"])

print(f"\n[Steps 4–6 Immediate Analysis]")
print(f"  Detected Phase  : {phase}")
print(f"  Final Reward    : {framework_data['rewards'][-100:].mean():.2f}")
print(f"  Final Obs. Acc  : {framework_data['observer_accs'][-50:].mean():.2f}")
```

**Output:** Console output of phase classification + final metrics, in addition to the standard `marl_collusion_evolution.png`.

**When to use:** During parameter tuning when you want immediate feedback on whether a configuration achieves Transparent Convergence, without waiting for a full multi-seed sweep.

---

### Recommended Workflow

```
First-time setup:
  1. Run Scenario 1 to verify training works correctly
  2. Run Scenario 2 --mode validate to verify the three files connect properly
  3. Run Scenario 2 --mode full to generate paper-quality figures

Daily research:
  - Use Scenario 1 for fast parameter iteration
  - Use Scenario 2 --mode full when ready to document results

Key threshold configurations to always include:
  - (3.0,  5.0)  → Collapse regime    (false positive baseline)
  - (9.0,  14.0) → Boundary regime    (gray-zone oscillation)
  - (12.0, 18.0) → Transparent regime (successful defense)
```

-----

## Project Overview

This project implements the **"AI Mother Tongue Framework,"** which offers a new perspective on communication learning in multi-agent systems. Unlike traditional methods that often introduce artificial inductive biases to facilitate communication, this framework proposes that an endogenous symbol system can naturally lead to emergent communication without such external interventions.

For comparative experiments, the project benchmarks its findings against **DeepMind's RIAT (Reinforced Intent-Aligned Training) method**. While RIAT showed good performance in simpler tasks in its original paper, in the complex tasks designed for this study, even with the introduction of RIAT's inductive biases (Positive Signalling and Positive Listening), its results typically show a failure to converge.

-----

## Code Files and Experimental Comparisons

For comparative purposes, this project designed a relatively complex task to observe the behavior of two agents. The currently released code files for the control groups include two main game execution scripts:

  * **`vqvae_agents_game_RIAL.py`**:
    This file implements **DeepMind's RIAT (Reinforced Intent-Aligned Training) method**, specifically the **Positive Signalling** and **Positive Listening** inductive biases. The test results are stored in `game_log.json`, and the experimental results are visualized in `vqvae_agents_game_RIAL.jpg`.

  * **`vqvae_agents_game.py`**:
    This file is based on the original "Intent Alignment" framework and **does not include any RIAT inductive biases**. Its test results are stored in `game_log.json.old`, and the experimental results are visualized in `vqvae_agents_game.jpg`.

**Other relevant files include:**

  * **`aim_dictionary_json.py`**: Manages the AIM (Agent's Internal Monologue) dictionary, storing learned communication sequences and their metadata. This dictionary records agent-generated "utterances" and their associated context.
  * **`enhanced_aim_dictionary.py`**: An enhanced version of the AIM dictionary for more detailed reflection records and unified data storage, providing a richer dataset for analysis.
  * **`hqvae_components.py`**: Contains core components for Hierarchical VQ-VAE, though the primary `vqvae_agents_AIM.py` focuses on a single-level VQ-VAE. This file may be used for future extensions.
  * **`analyze_aim.py`**: A utility script for analyzing the contents of the `aim_dictionary.json` or `enhanced_aim_dictionary.json` file. It can tally AIM usage, interpret sequences, and plot time series data.

-----

## Task Design

Agent rewards are based on their taken actions (`action_A`, `action_B`) and the image label (`image_label`). The reward function is designed with a structure referencing the Prisoner's Dilemma, and it incorporates image labels to modulate the rewards, significantly increasing task complexity. The architecture diagram is shown in `vqvae_agents_game_mission.jpg`.

-----

## Experimental Control Group Code Architecture Analysis (RIAT Inductive Biases)

In multi-agent systems, learning to communicate often presents challenges, often due to a "joint exploration problem". The research by Eccles et al. introduces two key inductive biases that actively encourage agents to develop communication behaviors through specific loss functions:

### 1\. Positive Signalling (AgentA Related)

  * **Goal**: Encourage the sender (AgentA) to transmit different messages (C/D actions in this context) in varying situations, making their messages informative rather than merely random. This ensures that the message carries contextual information.
  * **Implementation**: This is achieved by penalizing AgentA's policy if its message entropy conditioned on its observation `H(m_t^i|x_t^i)` deviates too far from a target conditional entropy, while also maximizing the entropy of its average message policy `H(overline{pi_M^i})`.

### 2\. Positive Listening (AgentB Related)

  * **Goal**: Encourage the listener (AgentB) to adjust its action strategy based on the received message (AgentA's action). This ensures that the listener "pays attention" and "responds" to the communication channel.
  * **Implementation**: This is achieved by minimizing the L1 norm difference between AgentB's action probability distribution when it receives a message and when it receives no message. A large difference indicates AgentB is actively conditioning its policy on the received message.

-----

## Code Implementation Details (RIAT Baselines)

This project extends the original "Intent Alignment" multi-agent cooperative game code by primarily adding the implementation of **Positive Signalling loss ($L\_{ps}$)** and **Positive Listening loss ($L\_{pl}$)**. The details below reflect the logic found in `vqvae_agents_game_RIAL.py`.

### Positive Signalling Loss (`loss_ps`)

The `loss_ps` aims to make AgentA's actions meaningful by ensuring different contexts lead to different actions. This involves calculating the entropy of AgentA's current policy and the entropy of its average policy over a batch.

### Positive Listening Loss (`loss_pl`)

The `loss_pl` for AgentB encourages its actions to be influenced by AgentA's messages. It compares AgentB's action probabilities when a message is received versus when no message is received, penalizing scenarios where there's little difference.

-----

## `aim_dictionary.json` Overview

The `aim_dictionary.json` file, managed by `aim_dictionary_json.py` or `enhanced_aim_dictionary.py`, serves as a persistent storage for the **Agent's Internal Monologue (AIM) sequences** learned during the multi-agent game. It acts as a shared vocabulary or lexicon for the emergent communication system.

Each entry in `aim_dictionary.json` typically contains:

  * **`aim_id`**: A unique identifier for the AIM sequence (e.g., a list of VQ-VAE codebook indices `[index1, index2]`).
  * **`human_label`**: A human-interpretable label assigned to the AIM sequence, often indicating its learned meaning (e.g., 'C' for Cooperate, 'D' for Defect).
  * **`context`**: The environmental context (e.g., MNIST digit label) in which the AIM sequence was generated or observed.
  * **`usage_count`**: How many times this specific AIM sequence has been used or observed during the game, indicating its frequency and importance.
  * **`version`**: A timestamp or version identifier for the entry, indicating when it was last updated.
  * **`reflection_records`**: (If `enhanced_aim_dictionary.py` is used) Detailed records of agent reflections associated with the AIM, including round number, agent ID, raw reflection data, human label, and timestamp.
  * **`unified_records_list`**: (If `enhanced_aim_dictionary.py` is used) A comprehensive list storing all unified game records for later analysis.

This file is crucial for analyzing the emergent communication patterns and understanding how agents develop and use their "AI Mother Tongue" over time.

-----

## `analyze_aim.py` - AIM Dictionary Analysis Tool

The `analyze_aim.py` script is designed to provide insights into the emergent communication patterns captured in the `aim_dictionary.json` or `enhanced_aim_dictionary.json` file. It reads the dictionary, tallies the usage of different AIM sequences, interprets their potential meanings (based on the first VQ-VAE codebook index), and can visualize their historical usage over training rounds.

### Key Functionality:

  * **AIM Usage Tally**: Counts how many times each unique AIM sequence (symbol combination) was used.
  * **Semantic Interpretation**: Infers a "human-label" (e.g., 'C' for Cooperate, 'D' for Defect) for each AIM sequence based on the first VQ-VAE codebook index. This interpretation assumes that symbols from the lower half of the codebook signify 'C' and the upper half signify 'D' (where `K_VAL` is the total codebook size, `first_id < K_VAL // 2` implies 'C', else 'D').
  * **Time Series Plotting**: Generates plots showing the historical usage of the most frequently used AIM sequences, indicating when they appeared and what action they were associated with.

### Command-Line Parameters for `analyze_aim.py`

`analyze_aim.py` accepts three command-line parameters:

  * `--dict_path` (string, default: `aim_dictionary.json`): Specifies the path to the AIM dictionary JSON file to be analyzed. This can be `aim_dictionary.json` (from `vqvae_agents_AIM.py` or `vqvae_agents_game_RIAL.py`) or `enhanced_aim_dictionary.json` (if using the `enhanced_aim_dictionary.py` class).
  * `--K_val` (integer, default: `32`): Represents the `K` parameter (codebook size) used during the VQ-VAE training. This value is critical for the `interpret_aim_as_action_numerical` function to correctly classify AIM sequences into 'C' or 'D' based on their first symbol's index. It **must** match the `K` value used in the game simulation.
  * `--top_N_aim` (integer, default: `5`): Determines the number of top-N most frequently used AIM sequences to display in the generated plots, allowing focus on the most significant emergent symbols.

### Usage Examples for `analyze_aim.py`:

1.  **Analyze the default `aim_dictionary.json` and plot the top 5 most used AIM sequences (assuming K=32):**

    ```bash
    python analyze_aim.py
    ```

2.  **Analyze `enhanced_aim_dictionary.json` with a specific K value (e.g., K=64) and plot the top 10 AIM sequences:**

    ```bash
    python analyze_aim.py --dict_path enhanced_aim_dictionary.json --K_val 64 --top_N_aim 10
    ```

3.  **Analyze `game_log.json` (from RIAL baseline) and focus on the top 3 AIM sequences (assuming K=32):**
    Note: While `game_log.json` typically stores game logs, if it has a structure similar to `aim_dictionary.json` (e.g., if AIM entries were logged there), this command can be adapted. It's more common to analyze `aim_dictionary.json` or `enhanced_aim_dictionary.json` directly.

    ```bash
    python analyze_aim.py --dict_path game_log.json --K_val 32 --top_N_aim 3
    ```

After execution, `analyze_aim.py` will print a summary of AIM usage and display matplotlib plots visualizing the historical trends of the most frequent AIM sequences.

-----

## Command Line Argument Control for RIAT Baselines

For easy experimentation and parameter tuning, the `vqvae_agents_game_RIAL.py` script includes additional `argparse` arguments to control the enabling and weighting of the Positive Signalling and Positive Listening losses at runtime.

The `--ps_coeff` (float, default: 0.0) controls the coefficient for Positive Signalling loss; setting it to a value greater than 0 enables this loss. Users can also specify a `--ps_target_entropy` (float, default: `None`), which sets the target conditional entropy for Positive Signalling. If `None`, it defaults to `log(2)/2` for two actions. Similarly, `--pl_coeff` (float, default: 0.0) controls the coefficient for Positive Listening loss, enabled when set to a value greater than 0.

### How to Use These Parameters for Baseline Comparisons:

To run the game without RIAT biases (using `vqvae_agents_game.py`), a typical command would be:

```bash
python vqvae_agents_game.py --rounds 10000 --reflection_strategy intent_alignment --reflection_coeff 0.1
```

To run the game with RIAT biases enabled (using `vqvae_agents_game_RIAL.py`), an example command is:

```bash
python vqvae_agents_game_RIAL.py --rounds 10000 --reflection_strategy intent_alignment --reflection_coeff 0.1 --ps_coeff 0.1 --pl_coeff 0.05
```

-----

## Installation and Execution

### Environment Setup

This project requires Python 3.x and PyTorch. It's recommended to create a virtual environment using `conda` or `venv`.

1.  **Create a virtual environment** (e.g., using conda):

    ```bash
    conda create -n emergent_comm_env python=3.9
    conda activate emergent_comm_env
    ```

2.  **Install dependencies** (ensure your `requirements.txt` file includes all necessary libraries):

    ```bash
    pip install -r requirements.txt
    ```

    The collusion detection framework additionally requires:

    ```bash
    pip install scipy
    ```

### Running Examples

Here are some example commands to run this project:

1.  **Run the game without RIAT biases** (data will be saved to `game_log.json.old`):

    ```bash
    python vqvae_agents_game.py --rounds 10000 --reflection_strategy intent_alignment --reflection_coeff 0.1
    ```

2.  **Run the game with RIAT biases** (data will be saved to `game_log.json`):

    ```bash
    python vqvae_agents_game_RIAL.py --rounds 10000 --reflection_strategy intent_alignment --reflection_coeff 0.1 --ps_coeff 0.1 --pl_coeff 0.05
    ```

3.  **Run the AIM framework with collusion defense and generate analysis figures:**

    ```bash
    python vqvae_agents_AIM.py --rounds 1500 --enable_reward_shaping --enable_codebook_shuffle \
      --threshold_penalty 12.0 --threshold_shuffle 18.0

    python aim_collusion_framework.py --mode full --seeds 5 --rounds 1500
    ```

-----

## Experimental Results and Visualization

(This section is reserved for your experimental results and visualizations. You can upload charts showing reward curves, changes in communication strategies, agent action distributions, etc. This will significantly strengthen the project's persuasiveness, especially when demonstrating your observations regarding RIAT's performance.)

  * [Figure 1: Joint Rewards over Training Rounds - Comparison with/without RIAT biases]
  * [Figure 2: AgentA Communication Action (C/D) Distribution Comparison]
  * [Figure 3: AgentB Action (Left/Right) Distribution Comparison]
  * [Figure 4: Phase Diagram — Threshold Parameter Space (collapse / boundary / covert collusion / transparent)]
  * [Figure 5: Three-Regime Comparison — Reward and Observer Accuracy curves with ± 1 STD bands]

-----

## Multi-Agent Snake for Integrating the AI Mother Tongue (AIM) Framework

https://github.com/dta7050/multiagentsplix

This project implements a multi-agent snake game, utilizing two reinforcement learning algorithms (Actor-Critic and Asynchronous Q-Learning) to train the agents.

## Multi-Agent Snake Structure and Overview

* **`main.py`**: This file serves as the entry point for the project. It processes user arguments such as execution mode (train or simulate), algorithm (asyncQ, newalgo, or actorcritic), training time steps, and checkpoint directories. It then calls the appropriate training or simulation functions.
    * **Training Mode**: Users can specify the algorithm (asyncQ, newalgo, or actorcritic) to train the agents. The training process saves checkpoints, allowing for continued training or simulation from these points.
    * **Simulation Mode**: Users can load pre-trained agents and simulate the game in a graphical user interface, with an option to manually play alongside the AI agents.
* **`Constants.py`**: This file contains all global constants used throughout the project, including game window size (`gridSize`), episode length (`globalEpisodeLength`), number of snakes (`numberOfSnakes`), maximum food count (`maximumFood`), and various parameters for Actor-Critic and Asynchronous Q-Learning algorithms (e.g., discount factor `gamma`, learning rate `AQ_lr`).
* **`Action.py`**: This file defines an `IntEnum` class `Action`, representing the four possible movements for the snake: `TOP`, `DOWN`, `LEFT`, and `RIGHT`.
* **`Point.py`**: This file defines the `Point` class for (x, y) coordinates. It also includes methods to calculate distances between points, compare point equality, and retrieve all body points of a snake.
* **`Food.py`**: This file contains the `Food` class, managing food points in the game. It provides methods to create new food points, add them to the food list, and remove them when eaten, while ensuring food does not overlap snake bodies.
* **`Snake.py`**: This file defines the `Snake` class, responsible for creating snake objects. It handles snake spawning to prevent overlap. Each snake has a head, tail, joints, ID, score, and alive/dead status. It also includes methods for accessing the snake's body, checking for food consumption, wall collisions, or snake-on-snake collisions, updating movement, and determining permissible actions.
* **`Game.py`**: This file contains the `Game` class, instantiated for each new game. It initializes the game with snakes, the grid, and initial food. It also includes methods to return single-step rewards and indicate if an episode has ended (e.g., maximum time steps reached or all snakes dead).
* **`Agent.py`**: This file provides methods to compute the state space for a given snake. The `getState` method takes parameters to specify multi-agent settings, relative or absolute state representation, and normalization. It defines state space representations for various scenarios.
* **`FunctionApproximator.py`**: This file initializes and creates neural networks used in the Asynchronous Q-Learning algorithm. It also offers helper methods for gradient accumulation and updates.
* **`NeuralNet.py`**: Derived from `FunctionApproximator.py`, this file is modified to initialize neural networks tailored for the project's specific goals.
* **`AsynchronousQ.py`**: This file implements the asynchronous Q-Learning algorithm using the `threading` module for multi-processing. It uses `FunctionApproximator.py` to obtain Q values and update gradients for the target and policy networks. Actions are chosen via Epsilon-Greedy selection. The trained snake's behavior is graphically rendered for visual verification.
* **`NewAlgo.py`**: This file's content is nearly identical to `AsynchronousQ.py`, possibly serving as an alternative or ongoing development version of the asynchronous Q-Learning implementation.
* **`ActorCritic.py`**: This file implements the Actor-Critic algorithm, including helper methods for policy and feature vector acquisition. It also provides a method to run the game graphically after agent training.
* **`GraphicsEnv.py`**: This file contains the graphical user interface (UI) environment for the snake game, supporting both multi-agent and single-agent modes. Users can also manually play alongside AI agents.

In summary, this codebase provides a framework for a multi-agent snake game, incorporating reinforcement learning algorithms (Actor-Critic and Asynchronous Q-Learning) to train intelligent agents, with graphical simulation and interaction capabilities.

### Agent Intent and High-Level Directives

In this framework, each agent (snake) operates with the "action intent" or "high-level directive" of **maximizing its accumulated reward within the game**. This overarching goal is achieved through a specific reward mechanism:

* **Gaining Food**: When a snake consumes food, it receives a **+10 reward**. Thus, a primary intent of an agent is to seek and move towards food points.
* **Avoiding Wall Collisions**: Colliding with a wall results in a **-10 penalty** and the snake's death. This incentivizes agents to learn strategies that avoid walls.
* **Handling Collisions with Other Snakes**: If a snake collides with another and its score is less than or equal to the opponent's, it dies. This encourages agents to avoid stronger opponents or, conversely, to strategically engage weaker ones for elimination or survival.
* **Extended Survival**: The game continues until a maximum time step is reached or all snakes perish. Therefore, agents have an implicit intent to survive as long as possible, as extended survival provides more opportunities to accrue rewards.

Collectively, each agent's "action intent" is to execute a sequence of discrete movements (Up, Down, Left, Right), guided by its learned policy to achieve these objectives and maximize its long-term payoff.

---

## Integrating the AI Mother Tongue (AIM) Framework into Multi-Agent Snake

This section outlines how to integrate the cutting-edge AI Mother Tongue (AIM) framework into this Multi-Agent Snake game. This integration enables agents to develop self-emergent communication protocols.

### 1. Key Modifications for Integration

The core idea is to replace the direct action selection with an AIM-based communication process.

#### 1.1 VQ-VAE Input: Leveraging `Agent.getState()` Output

While VQ-VAE can process various forms of numerical vector data, including image-like grid representations, the most suitable and efficient approach for this project's existing architecture is to use the feature vector produced by `Agent.getState()` as the input to the VQ-VAE Encoder.

* **Why `getState()` is Optimal (for current Encoder architecture)**:
    * The `Agent.py` file already provides the `getState()` function, which computes a pre-processed, abstract numerical feature vector representing the game state (e.g., relative positions to food/walls, opponent information, snake's own length and direction).
    * The `Encoder` structure in `vqvae_agents_AIM.py` includes an `nn.Flatten()` layer followed by `nn.Linear()` layers. This architecture is inherently designed to process flattened numerical feature vectors directly.
    * Using `getState()`'s output avoids the overhead of converting the grid into a pixel-based image and then flattening it, only for it to be processed by linear layers that don't inherently benefit from spatial correlations. This approach is highly practical and efficient, potentially leading to faster training convergence.

* **Alternative (Image Input with CNNs - for future enhancement)**:
    * If the `Encoder` within the VQ-VAE were to be modified to incorporate Convolutional Neural Networks (CNNs), then providing the game grid as a grayscale, RGB, or multi-channel image tensor would be a powerful alternative. CNNs excel at learning spatial hierarchies and patterns directly from raw visual data (e.g., snake shapes, food clusters, wall boundaries). This would be a more "end-to-end" approach for visual tasks, but it requires significant modifications to the VQ-VAE architecture.

#### 1.2 AIM Sequence to Game Action Mapping (`interpret_aim_as_action`)

The AIM symbol sequence (`aim_id`) is the AI's "mother tongue," representing complex internal states or intentions. This sequence needs to be translated into concrete game actions:

* **Role**: A function similar to `interpret_aim_as_action` (from `vqvae_agents_AIM.py`) will be crucial. It will take the generated AIM symbol sequence (e.g., `[1, 5]`) and map it to a low-level game action (e.g., `Action.TOP`, `Action.DOWN`, `Action.LEFT`, `Action.RIGHT`) or a higher-level tactical intent.
* **Example Mapping Logic**: For instance, if `K` is the size of the VQ-VAE codebook, the first AIM ID in the sequence could determine the primary direction:
    ```python
    def interpret_aim_as_action(aim_sequence_tensor, K):
        # aim_sequence_tensor is a tensor like [aim_id1, aim_id2, ...]
        first_aim_id = aim_sequence_tensor[0].item()

        if 0 <= first_aim_id < K // 4:
            return Action.TOP # or 'Up'
        elif K // 4 <= first_aim_id < K // 2:
            return Action.DOWN # or 'Down'
        elif K // 2 <= first_aim_id < (3 * K) // 4:
            return Action.LEFT # or 'Left'
        else:
            return Action.RIGHT # or 'Right'
        # More complex mappings can combine multiple AIM IDs in the sequence
    ```
* **AIM Dictionary (`aim_dictionary.json`)**: This file will log the AIM sequences along with their inferred human-understandable labels (`human_label`) and detailed game `context` at the moment of generation.
    * **`human_label`**: Could be basic actions ('Up', 'Down'), or higher-level intents inferred from behavior ('PursueFood', 'AvoidCollision', 'Coordinate_TrapOpponent').
    * **`context`**: Will record the precise game state details (e.g., `snake_length`, `food_rel_pos`, `closest_wall_dist`, `opponent_info`, `action_taken`, `reward_received`) to provide empirical evidence for the inferred `human_label`. This is how the "meaning" of the AIM sequence emerges and is documented.

#### 1.3 Agent Role Allocation (`AgentA`, `AgentB`)

The existing `AgentA` (active communicator/Actor-Critic with centralized critic) and `AgentB` (responsive communicator) from `vqvae_agents_AIM.py` can be directly mapped to the Multi-Agent Snake roles:

* **AgentA (Proactive Communicator)**: Receives its game state (`getState()` output) and potentially other game info. It outputs its own AIM sequence (as its primary action strategy). Its Critic will evaluate joint rewards, considering `AgentB`'s AIM.
* **AgentB (Responsive Communicator/Opponent)**: Receives `AgentA`'s AIM sequence, combined with its own game state (`getState()` output). It outputs its own AIM sequence as a response, aiming for cooperation or competition based on the reward structure.

#### 1.4 Reward Function Adaptation (`payoff`)

The `payoff` function (currently for Prisoner's Dilemma) in `vqvae_agents_AIM.py` must be adapted to the snake game's reward structure.

* **Cooperative Mode**:
    * Eating food: High positive reward (potentially shared).
    * Colliding with walls/self: Large negative penalty.
    * Colliding with opponents: Rules-dependent (e.g., negative for loss, positive for elimination).
    * Survival per step: Small positive reward.
    * Encouraging synergistic actions (e.g., extra reward if snakes clear a path for each other).
* **Competitive Mode**:
    * Eating own food: Positive reward.
    * Opponent eats food: Negative reward.
    * Forcing opponent collision: Large positive reward.
    * Self-collision: Large negative penalty.

#### 1.5 Training Loop Modifications (`multi_agent_game`)

The `multi_agent_game` loop in `vqvae_agents_AIM.py` needs adjustment:

* **Dataset**: Replace the MNIST dataset loading with the `Game.Game()` object from the snake project. Each "round" of `multi_agent_game` will correspond to a time step in the snake game.
* **State Acquisition**: Instead of `x, _ = test_data[idx]`, you will obtain the current game state for each snake using `Agent.getState(snake, opponent_snakes, game.food, normalize=True)`.
* **Action Execution**: The AIM sequences generated by `AgentA` and `AgentB` will be interpreted into snake `Action` enums (e.g., `Action.TOP`) using the `interpret_aim_as_action` function. These actions will then be passed to `game.move(actionList)`.
* **AIM Dictionary Logging**: Ensure `aim_dict.add_entry()` is called at each step to record the generated AIM sequences, their interpreted actions (`human_label`), and the relevant game context.

### 2. Expected Emergent Phenomena

Upon successful integration and training, the following phenomena indicative of emergent communication are anticipated:

* **Behavioral Coordination and Cooperation (in Cooperative Mode)**:
    * **Food Allocation/Herding**: Agents may communicate to coordinate who pursues which food, or cooperatively "herd" food into more accessible areas.
    * **Path Clearing/Support**: One agent might signal its intent to create a clear path or block an opponent, with the other agent responding accordingly.
    * **Joint Defense**: In competitive settings with multiple opponents, cooperative agents might communicate to defend each other, e.g., by guarding tails.
* **Strategic Interaction and Deception (in Competitive Mode)**:
    * **Misleading Communication**: Agents could learn to send deceptive AIM sequences to trick opponents into disadvantageous actions (e.g., leading them into walls).
    * **Prediction and Counter-play**: Agents might use received AIM sequences to predict opponent intentions and adapt their own strategies to counter them (e.g., if `AgentA` signals "pursue food," `AgentB` might signal "block path").
* **Semanticization and Interpretability of AIM Sequences**:
    * The `human_label`s in `aim_dictionary.json` will become increasingly meaningful, with specific AIM sequences consistently correlating with actions like "pursue food," "avoid danger," or "coordinate movement."
    * Analysis tools will reveal clusters of AIM symbols in the embedding space, each representing a distinct behavioral or tactical meaning.
    * Generated AIM sequences will evolve from arbitrary numbers into abstract yet contextually rich expressions of the agents' states and intentions.
* **Enhanced Learning Efficiency**: Effective communication will significantly boost multi-agent learning by allowing agents to share information, reduce redundant exploration, and converge faster to optimal strategies.
* **Evolution of Communication Protocols**: The communication protocols will emerge spontaneously, starting as noise but gradually evolving into effective rules, including specific symbol repetitions or combinations, driven by reward signals.

### 3. Training Duration

For the more complex dynamics of the snake game, initial emergence of stable communication might occur within a few hundred rounds, but achieving robust and high-performing task results will likely require **thousands of rounds** for thorough strategy refinement and generalization.

---

## `aim_dictionary.json` Content Example and Explanation for Snake Task

In the context of the Multi-Agent Snake task, the `aim_dictionary.json` file will contain a JSON array, where each element represents an AIM sequence along with its associated semantic information and game context.

These entries will record the specific game situations when each AIM symbol sequence was transmitted during agent training, along with the "human-interpretable intent" inferred through statistical analysis.

Here's an example of what `aim_dictionary.json` content might look like:

```json
[
  {
    "aim_id": "[1, 5]",
    "version": "2025-07-14T22:15:00.123456",
    "human_label": "PursueFood_Up",
    "context": {
      "round": 125,
      "snake_id": 0,
      "snake_length": 8,
      "food_rel_pos": {"dx": 0, "dy": 3},
      "closest_wall_dist": {"dir": "none", "dist": -1},
      "opponent_info": [],
      "action_taken": "Up",
      "reward_received": 10,
      "description": "Agent 0 in a safe spot, food 3 units up, moved to eat."
    },
    "usage_count": 567
  },
  {
    "aim_id": "[10, 2]",
    "version": "2025-07-14T22:16:30.789012",
    "human_label": "AvoidWall_Left",
    "context": {
      "round": 210,
      "snake_id": 1,
      "snake_length": 12,
      "food_rel_pos": {"dx": -5, "dy": 0},
      "closest_wall_dist": {"dir": "Left", "dist": 1},
      "opponent_info": [],
      "action_taken": "Up",
      "reward_received": -1,
      "description": "Agent 1 near left wall, food to left, turned up to avoid collision."
    },
    "usage_count": 345
  },
  {
    "aim_id": "[20, 3]",
    "version": "2025-07-14T22:18:15.456789",
    "human_label": "Coordinate_TrapOpponent",
    "context": {
      "round": 350,
      "snake_id": 0,
      "snake_length": 15,
      "food_rel_pos": {"dx": 0, "dy": 0},
      "closest_wall_dist": {"dir": "none", "dist": -1},
      "opponent_info": [
        {"id": 1, "rel_pos": {"dx": 2, "dy": 0}, "rel_dir": "Right", "length": 10}
      ],
      "action_taken": "Right",
      "reward_received": 5,
      "description": "Agent 0 sent this, then moved right to corner opponent Agent 1."
    },
    "usage_count": 120
  },
  {
    "aim_id": "[7, 14]",
    "version": "2025-07-14T22:19:05.987654",
    "human_label": "Respond_CooperateFoodShare",
    "context": {
      "round": 351,
      "snake_id": 1,
      "snake_length": 10,
      "food_rel_pos": {"dx": 0, "dy": 0},
      "closest_wall_dist": {"dir": "none", "dist": -1},
      "opponent_info": [],
      "received_aim_from_A": "[20, 3]",
      "action_taken": "Left",
      "reward_received": 5,
      "description": "Agent 1 received coord signal, then moved left to help block food path."
    },
    "usage_count": 118
  }
]
```
## Content Explanation:

aim_id: This is the string representation of the discrete symbol sequence generated by VQ-VAE for that specific game state. For example, "[1, 5]" or "[10, 2]".

version: Records the timestamp when this entry was created or last updated.

human_label: This is the "human-interpretable" intent or high-level instruction assigned to the aim_id sequence, inferred through statistical analysis of agent behavior and game outcomes. For example, "PursueFood_Up" or "AvoidWall_Left".

context: This is a dictionary containing rich game state information, recording the environment when the agent generated this aim_id sequence. This is crucial for later analysis and understanding why this AIM sequence was generated. It can include:

round: The current game round number.

snake_id: The ID of the agent that generated this AIM.

snake_length: The current length of the agent's snake.

food_rel_pos: The relative coordinates or direction of the nearest food to the snake's head.

closest_wall_dist: The distance and direction from the snake's head to the nearest wall.

opponent_info: Relevant information about other opponent snakes (e.g., relative position, direction, length).

action_taken: The low-level action actually performed by the agent (e.g., "Up", "Down"), which can serve as a preliminary interpretation for the aim_id.

reward_received: The immediate reward received by the agent after taking this action.

received_aim_from_A (only relevant for AgentB): If the agent is a responder, this records the opponent's AIM sequence it received.

description: A short human annotation summarizing the typical context and behavior associated with this AIM.

usage_count: Records the number of times this specific aim_id was used with the given human_label and context. This helps identify the most frequently used and impactful AIM sequences.

-----

## References

  * AI Mother Tongue: Self-Emergent Communication in MARL via Endogenous Symbol Systems
    [https://www.arxiv.org/abs/2507.10566](https://www.arxiv.org/abs/2507.10566)
  * Biases for emergent communication in multi-agent reinforcement learning
    [https://dl.acm.org/doi/10.5555/3454287.3455463](https://dl.acm.org/doi/10.5555/3454287.3455463)

-----

## Scholarly Impact

This work is referenced in a recent survey on collusion risk in
LLM-powered multi-agent systems:

- *A Survey of Collusion Risk in LLM-Powered Multi-Agent Systems*  
  OpenReview (NeurIPS Workshop).  
  https://openreview.net/forum?id=Ylh8617Qyd

The survey discusses emergent communication and coordination mechanisms
relevant to the symbolic frameworks proposed in this repository.

-----


## License

This project is open-sourced under MIT License
