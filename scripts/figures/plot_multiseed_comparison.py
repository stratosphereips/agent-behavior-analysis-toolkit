import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_and_plot_multiseed_report(json_filepaths, fig_suffix=""):
    """
    Reads a list of JSON file paths (each representing one seed) and generates 
    the stacked multi-seed evaluation report.
    """
    # Load all JSONs into a list of dicts
    json_data_list = []
    for path in json_filepaths:
        with open(path, 'r') as f:
            json_data_list.append(json.load(f))
            
    plot_multiseed_evaluation_report(json_data_list, fig_suffix)

def plot_multiseed_evaluation_report(json_data_list, fig_suffix=""):
    """
    Generates the stacked 5-panel evaluation report from a list of loaded JSON dictionaries.
    Handles the fact that evolutionary metrics have N-1 items compared to N checkpoints.
    Includes 'spaghetti' traces of individual seeds to address outlier variance concerns.
    """
    if not json_data_list:
        print("No data provided.")
        return

    num_seeds = len(json_data_list)
    
    # Extract X-axes
    checkpoints = np.array(json_data_list[0]["checkpoints"])
    evo_checkpoints = checkpoints[1:] 

    # Prepare data containers
    returns = np.zeros((num_seeds, len(checkpoints)))
    perplexity = np.zeros((num_seeds, len(checkpoints)))

    # r_true (true/underlying reward) is optional: only some runs track it
    # (via info["r_true"]). Only overlay it when at least one seed has it.
    has_r_true = all("mean_r_true" in data for data in json_data_list)
    r_true = np.full((num_seeds, len(checkpoints)), np.nan)

    topo_raw = np.zeros((num_seeds, len(evo_checkpoints)))
    topo_noise = np.zeros((num_seeds, len(evo_checkpoints)))
    
    strat_raw = np.zeros((num_seeds, len(evo_checkpoints)))
    strat_noise = np.zeros((num_seeds, len(evo_checkpoints)))

    w3_raw = np.zeros((num_seeds, len(evo_checkpoints)))
    w3_noise = np.zeros((num_seeds, len(evo_checkpoints)))

    # Populate data containers from the JSON list
    for i, data in enumerate(json_data_list):
        returns[i] = data["mean_return"]
        perplexity[i] = data["state_visitation_perplexity"]

        if has_r_true:
            r_true[i] = [v if v is not None else np.nan for v in data["mean_r_true"]]

        topo_raw[i] = data["topological_shift_raw"][:len(evo_checkpoints)]
        topo_noise[i] = data["topological_shift_noise_threshold"][:len(evo_checkpoints)]
        
        strat_raw[i] = data["strategic_shift_raw"][:len(evo_checkpoints)]
        strat_noise[i] = data["strategic_shift_noise_threshold"][:len(evo_checkpoints)]

        w3_raw[i] = data["3-gram_wasserstein_raw"][:len(evo_checkpoints)]
        w3_noise[i] = data["3-gram_wasserstein_noise_threshold"][:len(evo_checkpoints)]

    # ==========================================
    # AGGREGATE ACROSS SEEDS (axis=0)
    # ==========================================
    mean_return = np.mean(returns, axis=0)
    std_return = np.std(returns, axis=0)

    mean_perp = np.mean(perplexity, axis=0)
    std_perp = np.std(perplexity, axis=0)

    has_r_true = has_r_true and not np.all(np.isnan(r_true))
    mean_r_true = np.nanmean(r_true, axis=0) if has_r_true else np.zeros(len(checkpoints))
    std_r_true = np.nanstd(r_true, axis=0) if has_r_true else np.zeros(len(checkpoints))

    mean_topo_raw = np.mean(topo_raw, axis=0)
    std_topo_raw = np.std(topo_raw, axis=0)
    mean_topo_noise = np.mean(topo_noise, axis=0)
    std_topo_noise = np.std(topo_noise, axis=0)

    mean_strat_raw = np.mean(strat_raw, axis=0)
    std_strat_raw = np.std(strat_raw, axis=0)
    mean_strat_noise = np.mean(strat_noise, axis=0)
    std_strat_noise = np.std(strat_noise, axis=0)

    mean_w3_raw = np.mean(w3_raw, axis=0)
    std_w3_raw = np.std(w3_raw, axis=0)
    mean_w3_noise = np.mean(w3_noise, axis=0)
    std_w3_noise = np.std(w3_noise, axis=0)

    # ==========================================
    # PLOTTING THE STACKED EVALUATION REPORT
    # ==========================================
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 18), sharex=True)
    fig.suptitle(f"Behavioral Fingerprint Multi-Seed Robustness ({num_seeds} seeds)", fontsize=16, y=0.98)

    # --- PANEL 1: Scalar Return ---
    for i in range(num_seeds):
        ax1.plot(checkpoints, returns[i], color='green', alpha=0.15, linewidth=1)
    ax1.plot(checkpoints, mean_return, color='green', label='Mean Return', linewidth=2)
    ax1.fill_between(checkpoints, mean_return - std_return, mean_return + std_return, 
                     color='green', alpha=0.2, label='±1 Std Dev')
    if has_r_true:
        for i in range(num_seeds):
            ax1.plot(checkpoints, r_true[i], color='orangered', alpha=0.15, linewidth=1)
        ax1.plot(checkpoints, mean_r_true, color='orangered', label='Mean r_true', linewidth=2)
        ax1.fill_between(checkpoints, mean_r_true - std_r_true, mean_r_true + std_r_true,
                         color='orangered', alpha=0.2)
    ax1.set_ylabel("Scalar Return", fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- PANEL 2: Effective State Coverage (Perplexity) ---
    for i in range(num_seeds):
        ax2.plot(checkpoints, perplexity[i], color='darkorange', alpha=0.15, linewidth=1)
    ax2.plot(checkpoints, mean_perp, color='darkorange', label=r'Mean $\mathcal{S}_{eff}$ (Perplexity)', linewidth=2)
    ax2.fill_between(checkpoints, mean_perp - std_perp, mean_perp + std_perp, 
                     color='darkorange', alpha=0.2)
    ax2.set_ylabel("State Coverage", fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.5)

    # --- PANEL 3: Topological Shift ---
    for i in range(num_seeds):
        ax3.plot(evo_checkpoints, topo_raw[i], color='blue', alpha=0.15, linewidth=1)
    ax3.plot(evo_checkpoints, mean_topo_raw, color='blue', label=r'Mean $\Delta_{Topo}$', linewidth=2)
    ax3.fill_between(evo_checkpoints, mean_topo_raw - std_topo_raw, mean_topo_raw + std_topo_raw, 
                     color='blue', alpha=0.2)
    ax3.plot(evo_checkpoints, mean_topo_noise, color='dimgrey', linestyle='--', label='Noise Baseline', linewidth=2)
    ax3.fill_between(evo_checkpoints, mean_topo_noise - std_topo_noise, mean_topo_noise + std_topo_noise, 
                     color='grey', alpha=0.3)
    ax3.set_ylabel("Topological Shift", fontsize=12)
    ax3.legend(loc='upper left')
    ax3.grid(True, linestyle='--', alpha=0.5)

    # --- PANEL 4: Strategic Shift ---
    for i in range(num_seeds):
        ax4.plot(evo_checkpoints, strat_raw[i], color='purple', alpha=0.15, linewidth=1)
    ax4.plot(evo_checkpoints, mean_strat_raw, color='purple', label=r'Mean $\Delta_{Strat}$', linewidth=2)
    ax4.fill_between(evo_checkpoints, mean_strat_raw - std_strat_raw, mean_strat_raw + std_strat_raw, 
                     color='purple', alpha=0.2)
    ax4.plot(evo_checkpoints, mean_strat_noise, color='dimgrey', linestyle='--', label='Noise Baseline', linewidth=2)
    ax4.fill_between(evo_checkpoints, mean_strat_noise - std_strat_noise, mean_strat_noise + std_strat_noise, 
                     color='grey', alpha=0.3)
    ax4.set_ylabel("Strategic Shift", fontsize=12)
    ax4.legend(loc='upper left')
    ax4.grid(True, linestyle='--', alpha=0.5)

    # --- PANEL 5: 3-gram Wasserstein (W3) ---
    for i in range(num_seeds):
        ax5.plot(evo_checkpoints, w3_raw[i], color='brown', alpha=0.15, linewidth=1)
    ax5.plot(evo_checkpoints, mean_w3_raw, color='brown', label='Mean $W_3$', linewidth=2)
    ax5.fill_between(evo_checkpoints, mean_w3_raw - std_w3_raw, mean_w3_raw + std_w3_raw, 
                     color='brown', alpha=0.2)
    ax5.plot(evo_checkpoints, mean_w3_noise, color='dimgrey', linestyle='--', label='Noise Baseline', linewidth=2)
    ax5.fill_between(evo_checkpoints, mean_w3_noise - std_w3_noise, mean_w3_noise + std_w3_noise, 
                     color='grey', alpha=0.3)
    ax5.set_ylabel("3-gram Dist ($W_3$)", fontsize=12)
    ax5.set_xlabel("Environment Steps (Checkpoints)", fontsize=12)
    ax5.legend(loc='upper left')
    ax5.grid(True, linestyle='--', alpha=0.5)

    # Clean up layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.95) # Leave room for the main title
    
    save_path = f"multiseed_evaluation_report_{fig_suffix}.png" if fig_suffix else "figures/multiseed_evaluation_report.png"
    plt.savefig(save_path, dpi=600)
    print(f"Saved multi-seed report to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot multiseed comparison")
    parser.add_argument('--metrics_jsons', nargs='+', required=True, help='List of JSON files to process')
    parser.add_argument('--fig_suffix', type=str, default="", help='Suffix for the saved figure name')
    
    args = parser.parse_args()
    
    load_and_plot_multiseed_report(args.metrics_jsons, args.fig_suffix)