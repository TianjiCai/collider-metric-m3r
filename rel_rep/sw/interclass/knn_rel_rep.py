import numpy as np
import ot_utils
import comparison_utils
from tqdm import tqdm
from itertools import combinations
import sys


def main():
    dataset = sys.argv[1]
    metric = sys.argv[2]
    anchor_per_class = int(sys.argv[3])

    local_project_path = ""

    with open("project_path.txt") as f:
        local_project_path = f.read()

    num_events_per_class = 0

    if dataset == "mini":
        num_events_per_class = 1000
    elif dataset == "medium":
        num_events_per_class = 4000
    elif dataset == "full":
        num_events_per_class = 10000
    else:
        raise ValueError("Dataset must be one of 'mini', 'medium', or 'full'.")

    test_aucs = np.zeros((8, 8))
    best_ks = np.zeros((8, 8))

    for i, j in tqdm(
        combinations(range(8), 2), total=len(list(combinations(range(8), 2)))
    ):
        self_matrix_1_file = (
            local_project_path
            + f"dist_mat_gen/{metric}/{dataset}_{metric}/{metric}_block_{i}_{i}.npy"
        )
        self_matrix_2_file = (
            local_project_path
            + f"dist_mat_gen/{metric}/{dataset}_{metric}/{metric}_block_{j}_{j}.npy"
        )
        off_diag_matrix_file = (
            local_project_path
            + f"dist_mat_gen/{metric}/{dataset}_{metric}/{metric}_block_{i}_{j}.npy"
        )

        big_matrix = ot_utils.combine_distance_blocks(
            [self_matrix_1_file, off_diag_matrix_file, self_matrix_2_file]
        )

        labels = np.repeat([0, 1], num_events_per_class)

        rel_reps = comparison_utils.relative_rep(
            big_matrix,
            labels,
            num_classes=2,
            anchors_per_class=anchor_per_class,
            seed=42,
        )

        neighbor_list = list(range(1, int(np.sqrt(num_events_per_class * 2) * 2), 2))

        test_auc, best_k = comparison_utils.knn_with_features(
            rel_reps, labels, neighbor_list
        )

        test_aucs[i, j] = test_auc
        best_ks[i, j] = best_k

    test_aucs = test_aucs + test_aucs.T
    best_ks = best_ks + best_ks.T

    output_file_name = f"knn_sw_rel_rep_{anchor_per_class}.npz"
    np.savez(file=output_file_name, test_aucs=test_aucs, best_ks=best_ks)


if __name__ == "__main__":
    main()
