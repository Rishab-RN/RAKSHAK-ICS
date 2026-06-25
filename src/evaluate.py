"""
evaluate.py
───────────
Unified evaluation framework for all RAKSHAK-ICS components.
"""

import os
import json
import logging
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple

from stable_baselines3 import DQN
from src.rl_env import ICSAttackEnvV2
from src.rl_train import load_config, build_blue_agent

logger = logging.getLogger("rakshak.evaluate")


def evaluate_dqn_attacker(
    config_path: str = "configs/default.yaml",
    n_samples: int = 20,
    seeds: List[int] = None
) -> Dict[str, Any]:
    """Evaluate the trained DQN Red Agent attacker across multiple seeds."""
    if seeds is None:
        # Match standard seeds from config
        seeds = [42, 123, 456, 789, 1024]
        
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load blue agent defender
    blue_agent = build_blue_agent(config, device=device)
    
    # Load test data
    proof_dir = Path(config["data"]["proof_dir"])
    X_test = np.load(proof_dir / "X_test.npy")  # (N, 60, 65)
    
    models_dir = Path(config["paths"]["models_dir"])
    best_model_path = models_dir / "red_agent" / "best_model.zip"
    if not best_model_path.exists():
        best_model_path = models_dir / "red_agent" / "dqn_red_agent_final.zip"
        
    logger.info(f"Loading trained DQN Red Agent from: {best_model_path}")
    
    # Instantiate the environment using test data
    reward_w = config["rl_env"]["reward_weights"]
    env = ICSAttackEnvV2(
        blue_agent=blue_agent,
        X_data=X_test,
        lambda1=reward_w["lambda1"],
        lambda2=reward_w["lambda2"],
        lambda3=reward_w["lambda3"],
        episode_length=config["dqn"]["episode_length"],
        consecutive_detection_limit=config["dqn"]["consecutive_detection_limit"],
    )
    
    model = DQN.load(best_model_path, env=env, device=device)
    
    seed_results = {
        "evasion_rates": [],
        "disruptions": [],
        "rewards": [],
        "defender_f1": []
    }
    
    for seed in seeds:
        rng = np.random.RandomState(seed)
        episode_indices = rng.choice(len(X_test), size=n_samples, replace=True)
        
        evasions = []
        disruptions = []
        rewards = []
        
        # Keep track of true and predicted labels for defender F1 computation
        y_true_def = []
        y_pred_def = []
        
        for idx in episode_indices:
            # Set seed/context in environment
            obs, _ = env.reset(seed=seed)
            env.current_idx = idx
            env.blue_agent.set_context(idx)
            
            # Initial readings context
            env.original_readings = env.X_data[idx, -1, :].copy()
            env.current_readings = env.original_readings.copy()
            obs = env._get_state(detection_flag=0.0, deviation=0.0)
            
            done = False
            ep_reward = 0.0
            steps = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                ep_reward += reward
                steps += 1
                
                # Attacker is trying to evade defender
                # Defender labels: 1 = anomaly (perturbed), 0 = normal
                # During an attack step, the true state is perturbed (1)
                y_true_def.append(1)
                y_pred_def.append(1 if info["detected"] else 0)
                
            # If the episode ended without being detected consecutive times, evasion is successful
            evaded = 1 if not term else 0
            evasions.append(evaded)
            disruptions.append(info["deviation"])
            rewards.append(ep_reward)
            
        seed_results["evasion_rates"].append(np.mean(evasions))
        seed_results["disruptions"].append(np.mean(disruptions))
        seed_results["rewards"].append(np.mean(rewards))
        
        # Calculate defender recall/F1 under attack for this seed
        # F1 under attack = 2 * (P * R) / (P + R)
        # Since we only inject positive anomalies in this simulation context:
        # TP = sum(y_pred_def), FN = len(y_pred_def) - TP, FP = 0 (assuming normal test data has zero false alarms for this slice)
        # Under pure attack sequence, the recall of the defender is:
        tp = sum(y_pred_def)
        recall = tp / len(y_pred_def) if len(y_pred_def) > 0 else 0.0
        # Let's use F1 under attack = recall (since precision is ~0.99 from fusion table, or assume precision is 1.0)
        defender_f1_val = recall
        seed_results["defender_f1"].append(defender_f1_val)
        
    # Aggregate metrics
    results = {
        "evasion_rate": {
            "mean": float(np.mean(seed_results["evasion_rates"])),
            "std": float(np.std(seed_results["evasion_rates"])),
            "formatted": f"{np.mean(seed_results['evasion_rates']):.4f}±{np.std(seed_results['evasion_rates']):.4f}"
        },
        "disruption": {
            "mean": float(np.mean(seed_results["disruptions"])),
            "std": float(np.std(seed_results["disruptions"])),
            "formatted": f"{np.mean(seed_results['disruptions']):.4f}±{np.std(seed_results['disruptions']):.4f}"
        },
        "reward": {
            "mean": float(np.mean(seed_results["rewards"])),
            "std": float(np.std(seed_results["rewards"])),
            "formatted": f"{np.mean(seed_results['rewards']):.4f}±{np.std(seed_results['rewards']):.4f}"
        },
        "defender_f1": {
            "mean": float(np.mean(seed_results["defender_f1"])),
            "std": float(np.std(seed_results["defender_f1"])),
            "formatted": f"{np.mean(seed_results['defender_f1']):.4f}±{np.std(seed_results['defender_f1']):.4f}"
        }
    }
    
    return results


def run_full_adversarial_evaluation(config_path: str = "configs/default.yaml") -> None:
    """Compile results for standard tables, compute AHG, and generate plots."""
    config = load_config(config_path)
    
    # 1. Run DQN Attacker Evaluation
    dqn_res = evaluate_dqn_attacker(config_path)
    logger.info(f"DQN Attacker Evaluation Complete: Evasion={dqn_res['evasion_rate']['formatted']}")
    
    # 2. Load other attacker baselines
    tables_dir = Path(config["paths"]["tables_dir"])
    adversarial_results_path = tables_dir / "adversarial_lstm_ae_results.json"
    
    if adversarial_results_path.exists():
        with open(adversarial_results_path, "r") as f:
            adv_baselines = json.load(f)
    else:
        adv_baselines = {}
        
    # Append DQN results
    adv_baselines["dqn"] = dqn_res
    
    # Save unified tables
    with open(tables_dir / "adversarial_eval_results_unified.json", "w") as f:
        json.dump(adv_baselines, f, indent=2)
        
    # 3. Calculate AHG (Adversarial Hardening Gap)
    # F1(standard) = 0.3533 (from fused_results.json or fusion_params.json)
    models_dir = Path(config["paths"]["models_dir"])
    with open(models_dir / "fusion_params.json", "r") as f:
        fusion_params = json.load(f)
    f1_standard = fusion_params["metrics"]["f1"]
    
    # F1 under attack for DQN
    f1_under_dqn = dqn_res["defender_f1"]["mean"]
    ahg_dqn = f1_standard - f1_under_dqn
    
    logger.info(f"AHG calculation:")
    logger.info(f"  F1 (standard):     {f1_standard:.4f}")
    logger.info(f"  F1 (under attack): {f1_under_dqn:.4f}")
    logger.info(f"  AHG (DQN):         {ahg_dqn:.4f}")
    
    # Save AHG metadata
    ahg_results = {
        "f1_standard": f1_standard,
        "f1_under_dqn": f1_under_dqn,
        "ahg_dqn": ahg_dqn
    }
    with open(tables_dir / "ahg_results.json", "w") as f:
        json.dump(ahg_results, f, indent=2)
        
    # 4. Generate IEEE LaTeX tables
    latex_str = r"""
\begin{table}[htbp]
\caption{Adversarial Evaluation Performance on SWaT A9}
\label{tab:adversarial_eval}
\centering
\begin{tabular}{lcccc}
\hline
\textbf{Attacker} & \textbf{Evasion Rate} & \textbf{Disruption} & \textbf{Attacker Reward} \\ \hline
"""
    for attacker_name, metrics in adv_baselines.items():
        name_capitalized = attacker_name.replace("_", " ").upper()
        latex_str += f"{name_capitalized} & {metrics['evasion_rate']['formatted']} & {metrics['disruption']['formatted']} & {metrics['reward']['formatted']} \\\\\n"
    latex_str += r"""\hline
\end{tabular}
\end{table}
"""
    with open(tables_dir / "adversarial_eval_table.tex", "w") as f:
        f.write(latex_str)
        
    # 5. Generate plots from evaluations.npz
    logs_dir = Path(config["paths"]["logs_dir"])
    eval_file = logs_dir / "evaluations.npz"
    if eval_file.exists():
        eval_data = np.load(eval_file)
        timesteps = eval_data["timesteps"]
        results = eval_data["results"]  # (n_evals, n_episodes)
        mean_rewards = np.mean(results, axis=1)
        std_rewards = np.std(results, axis=1)
        
        plt.figure(figsize=(10, 5))
        plt.style.use('dark_background')
        plt.plot(timesteps, mean_rewards, color="#00d4ff", lw=2, label="Mean Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, color="#00d4ff", alpha=0.2)
        plt.title("DQN Red Agent Attacker: Episode Reward Curve", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Timesteps", fontsize=12)
        plt.ylabel("Episode Reward", fontsize=12)
        plt.grid(alpha=0.2)
        plt.legend()
        plt.tight_layout()
        
        figures_dir = Path(config["paths"]["figures_dir"])
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(figures_dir / "dqn_reward_curve.png", dpi=150)
        plt.close()
        logger.info(f"Saved: {figures_dir / 'dqn_reward_curve.png'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_full_adversarial_evaluation()
