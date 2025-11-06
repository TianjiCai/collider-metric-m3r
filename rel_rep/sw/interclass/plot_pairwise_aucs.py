import matplotlib.pyplot as plt
import sys
import numpy as np

def main():
    npz_file_name = sys.argv[1]
    data_entry = sys.argv[2]
    
    file = np.load(npz_file_name)
    data = file[data_entry]
    n = data.shape[0]

    mask = np.triu(np.ones_like(data, dtype=bool), k=1)
    plot_data = np.where(mask, data, np.nan)

    vals = plot_data[~np.isnan(plot_data)]
    nonzero = vals[vals != 0]
    vmin, vmax = float(nonzero.min()), float(nonzero.max())

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color=(1,1,1,0))  # transparent for masked cells

    im = ax.imshow(plot_data, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"{i+1}" for i in range(n)], rotation=45, ha="right")
    ax.set_yticklabels([f"{i+1}" for i in range(n)])
    ax.set_title("Pairwise kNN AUCs, Joint Relative Space, 200 Anchors")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Value")

    for i in range(n):
        for j in range(i+1, n):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=12, color="w")

    plt.tight_layout()
    plt.savefig("pairwise_knn_aucs_100.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()