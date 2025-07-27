import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def scale_to_zero_one(x, x_min,x_max):
    return (x-x_min)/(x_max-x_min)

def plot_graph_metrics_per_checkpoint(metrics:dict, change_points:set, winrate:list, stopping_cp=None, fig_sufix=""):
        checkpoints = list(metrics.keys())
        metric_keys = list(next(iter(metrics.values())).keys())

        fig, ax = plt.subplots(figsize=(14, 8))

        for metric_key in metric_keys:
            metric_values = [metrics[cp][metric_key] for cp in checkpoints]
            min_metric = min(metric_values)
            max_metric = max(metric_values)
            if metric_key == "wasserstein":
                print("wasserstein", metric_values)
                if max(metric_values) > 1:
                    metric_values = [scale_to_zero_one(x,min_metric, max_metric) for x in metric_values]
            elif metric_key == "GED":
                print("GED", metric_values)
                if max(metric_values) > 1:
                    metric_values = [scale_to_zero_one(x,min_metric, max_metric) for x in metric_values]
            else:
                metric_values = [1-x for x in metric_values]
            ax.plot(np.array(checkpoints)+1, metric_values, label=metric_key, marker='o')
       
        # add wirate
        ax.plot(np.arange(0, len(winrate), step=1),winrate, label="Winrate", marker="x",linestyle = '--' )
        ax.set_xlabel('Checkpoint ID')
        ax.set_ylabel('Graph Distance Measurement(scaled to [0,1])')
        ax.set_title('Graph Distance Measurements per checkpoint')
        ax.set_xticks(np.arange(0, len(checkpoints)+1, step=1))
        ax.legend()
        ax.grid(True)
        all_cp = []
        for cp_list in change_points.values():
            all_cp += cp_list
        for cp, count in Counter(all_cp).items():
            if cp == len(checkpoints):
                continue
            print(cp, count,count/len(metric_keys), f"WR={winrate[cp+1]}")
            count = 0.1+(count/len(metric_keys))*(0.8-0.1)
            start = cp  # Start of the change point
            end = cp + 1    # End of the Change point
            plt.axvspan(start, end, color='orange', alpha=count)
        # for cp in change_points:
        #     if cp == len(checkpoints)-1:
        #         continue
        #     start = cp  # Start of the change point
        #     end = cp+1   # End of the Change point
        #     plt.axvspan(start, end, color='orange', alpha=0.4)
        if stopping_cp:
            ax.annotate("Stopping point", xy=(stopping_cp, winrate[stopping_cp]), 
            xytext=(stopping_cp, winrate[stopping_cp] + 0.2), arrowprops=dict(facecolor='green', arrowstyle="->"), fontsize=14)
        plt.tight_layout()
        plt.savefig(f"figures/graph_metrics_per_checkpoint_{fig_sufix}.png",dpi=600)