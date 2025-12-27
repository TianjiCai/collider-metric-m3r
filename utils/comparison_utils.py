import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from tqdm import tqdm
import utils.ot_utils
import itertools
# pyright: reportArgumentType=false

sqrt2 = np.sqrt(2)


def strict_jaccard_similarity(
    distance_matrix_1, distance_matrix_2, query_index, n_neighbors
):
    sorted_indices_1 = np.argsort(distance_matrix_1[query_index])
    sorted_indices_2 = np.argsort(distance_matrix_2[query_index])

    neighbors_1 = sorted_indices_1[1 : n_neighbors + 1]
    neighbors_2 = sorted_indices_2[1 : n_neighbors + 1]

    intersection = np.intersect1d(neighbors_1, neighbors_2, assume_unique=True)

    similarity = len(intersection) / n_neighbors

    # print("=== Query Index:", query_index, "===")
    # print("Neighbors in Matrix 1")
    # print("Neibors:", neighbors_1)
    # print()
    # print("Neighbors in Matrix 2")
    # print("Neibors:", neighbors_2)

    return similarity


def average_strict_jaccard(distance_matrix_1, distance_matrix_2, n_neighbors):
    length = distance_matrix_1.shape[0]

    similarities = np.zeros(length)

    for query_index in range(length):
        similarity = strict_jaccard_similarity(
            distance_matrix_1, distance_matrix_2, query_index, n_neighbors
        )
        similarities[query_index] = similarity

    return similarities.mean()


def strict_mrr(distance_matrix_1, distance_matrix_2, query_index, n_neighbors):
    length = distance_matrix_1.shape[0]
    neigh_1 = NearestNeighbors(n_neighbors=length, metric="precomputed")
    neigh_2 = NearestNeighbors(n_neighbors=length, metric="precomputed")

    neigh_1.fit(distance_matrix_1)
    neigh_2.fit(distance_matrix_2)

    row_1 = distance_matrix_1[query_index].reshape(1, -1)

    idxs_1 = neigh_1.kneighbors(
        row_1, n_neighbors=n_neighbors + 1, return_distance=False
    )
    indices = idxs_1[0][1:]

    reciprocal_ranks = np.zeros(n_neighbors)
    for i, ref in enumerate(indices):
        row_ref = distance_matrix_2[ref].reshape(1, -1)
        idxs_ref = neigh_2.kneighbors(row_ref, n_neighbors=length)

        reciprocal_rank = 1 / (np.where(idxs_ref == query_index)[0][0])
        reciprocal_ranks[i] = reciprocal_rank

    return reciprocal_ranks.mean()


def class_level_jaccard_similarity(
    distance_matrix_1, distance_matrix_2, labels, query_index, n_neighbors
):
    labels_1 = np.copy(labels)
    labels_2 = np.copy(labels)

    row1 = np.copy(distance_matrix_1[query_index])
    row2 = np.copy(distance_matrix_2[query_index])

    sorted_indices_1 = np.argsort(row1)
    sorted_indices_2 = np.argsort(row2)

    sorted_labels_1 = labels_1[sorted_indices_1]
    sorted_labels_2 = labels_2[sorted_indices_2]

    neighbors_labels_1 = sorted_labels_1[1 : n_neighbors + 1]
    neighbors_labels_2 = sorted_labels_2[1 : n_neighbors + 1]

    u1, c1 = np.unique(neighbors_labels_1, return_counts=True)
    u2, c2 = np.unique(neighbors_labels_2, return_counts=True)
    _, i1, i2 = np.intersect1d(u1, u2, assume_unique=True, return_indices=True)

    return np.minimum(c1[i1], c2[i2]).sum() / n_neighbors


def avg_class_jaccard(
    distance_matrix_1, distance_matrix_2, labels, target_class, n_neighbors
):
    target_class_indices = np.where(labels == target_class)[0]
    # print(target_class_indices)

    length = target_class_indices.shape[0]

    similarities = np.zeros(length)

    for i, query_index in enumerate(target_class_indices):
        similarity = class_level_jaccard_similarity(
            distance_matrix_1, distance_matrix_2, labels, query_index, n_neighbors
        )
        similarities[i] = similarity

    # print(similarities)

    return similarities.mean()


def relative_rep(distance_matrix_1, labels, num_classes, anchors_per_class, seed):
    np.random.seed(seed)

    labels_buckets = [np.where(labels == i)[0] for i in range(num_classes)]
    # print(labels_buckets[0].shape)
    # print(labels_buckets[1].shape)

    anchors_indices = np.zeros(anchors_per_class * num_classes, dtype=int)

    for i in range(num_classes):
        class_anchors = np.random.choice(
            labels_buckets[i], size=anchors_per_class, replace=False
        )
        anchors_indices[i * anchors_per_class : (i + 1) * anchors_per_class] = (
            class_anchors
        )

    # print(anchors_indices)
    rel_rep_1 = np.zeros((len(labels), anchors_per_class * num_classes))

    for i in range(len(labels)):
        rel_rep_1[i] = distance_matrix_1[i][anchors_indices]

    return rel_rep_1


def relative_rep_ref(events, ref_events):
    rel_rep = np.zeros((events.shape[0], ref_events.shape[0]))

    for i, j in tqdm(
        itertools.product(
            range(events.shape[0]),
            range(ref_events.shape[0]),
        ),
        total=events.shape[0] * ref_events.shape[0],
    ):
        rel_rep[i][j] = utils.ot_utils.get_W2_Kin(events[i], ref_events[j])

    return rel_rep


def relative_rep_parallel(events, ref_events, batch_size=1024):
    return utils.ot_utils.parallel_dist_mat_non_square(
        event_set_1=events,
        event_set_2=ref_events,
        dist_func=utils.ot_utils.get_W2_Kin,
        events_per_process=batch_size,
    )


def decision_ordering_sim(rel_rep_1, rel_rep_2):
    n = rel_rep_1.shape[0]
    i_ind, j_ind = np.triu_indices(n, k=1)

    dec_ord = 0

    for i, j in zip(i_ind, j_ind):
        dec_ord += decision_ordering(rel_rep_1, rel_rep_2, i, j)

    return dec_ord / len(i_ind)


def decision_ordering_sim_2(rel_rep_1, rel_rep_2):
    n = rel_rep_1.shape[0]
    i_ind, j_ind = np.triu_indices(n, k=1)

    dec_ord = 0

    for i, j in zip(i_ind, j_ind):
        dec_ord += decision_ordering_2(rel_rep_1, rel_rep_2, i, j)

    return dec_ord / len(i_ind)


def decision_ordering(rel_rep_1, rel_rep_2, event_1_ind, event_2_ind):
    event_1_rel_1 = rel_rep_1[event_1_ind]
    event_2_rel_1 = rel_rep_1[event_2_ind]

    event_1_rel_2 = rel_rep_2[event_1_ind]
    event_2_rel_2 = rel_rep_2[event_2_ind]

    return np.mean(
        np.heaviside(
            (event_1_rel_1 - event_2_rel_1) * (event_1_rel_2 - event_2_rel_2), 0.5
        )
    )


def decision_ordering_2(rel_rep_1, rel_rep_2, event_1_ind, event_2_ind):
    event_1_rel_1 = rel_rep_1[event_1_ind]
    event_2_rel_1 = rel_rep_1[event_2_ind]

    event_1_rel_2 = rel_rep_2[event_1_ind]
    event_2_rel_2 = rel_rep_2[event_2_ind]

    return np.heaviside(
        (np.mean(event_1_rel_1) - np.mean(event_2_rel_1))
        * (np.mean(event_1_rel_2) - np.mean(event_2_rel_2)),
        0.5,
    )


def cos_sim(vec_1: np.ndarray, vec_2: np.ndarray):
    return vec_1.dot(vec_2)


def eud_sim(vec_1: np.ndarray, vec_2: np.ndarray):
    return 1 - np.linalg.norm(vec_1 - vec_2) / sqrt2


def eud_sim_2(vec_1: np.ndarray, vec_2: np.ndarray):
    d = np.linalg.norm(vec_1 - vec_2)
    return 1 / (1 + d)


def tanimoto_sim(vec_1: np.ndarray, vec_2: np.ndarray):
    dot_prod = vec_1.dot(vec_2)
    return dot_prod / (vec_1.dot(vec_1) + vec_2.dot(vec_2) - dot_prod)


def eud_dist(vec_1: np.ndarray, vec_2: np.ndarray):
    d = np.linalg.norm(vec_1 - vec_2)
    return d


def knn_with_features(
    X: np.ndarray,
    y: np.ndarray,
    k_list: list[int],
    metric: str = "euclidean",
    return_auc_list: bool = False,
    random_state: int = 42,
):
    """
    Perform k-Nearest Neighbors (kNN) classification using Euclidean distance with
    an automatic 60/20/20 train/validation/test split.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.

    y : np.ndarray of shape (n_samples,)
        Binary class labels (0 or 1).

    k_list : list[int]
        List of k (neighbor counts) to evaluate during model selection.

    return_auc_list : bool, optional (default=False)
        If True, also returns the validation AUC scores for all k.

    random_state : int, optional (default=42)
        Random seed for reproducibility.

    Returns
    -------
    test_auc : float
        AUC score on the test set using the best-performing k.

    best_k : int
        The neighbor count (k) that achieved the highest validation AUC.

    val_auc_list : list[float], optional
        Validation AUCs for each k (only returned if `return_auc_list=True`).
    """

    # --- Shuffle dataset ---
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X))
    X, y = X[indices], y[indices]

    # --- Split sizes ---
    n_samples = len(X)
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)
    n_test = n_samples - n_train - n_val

    # --- Dataset splits ---
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_trainval, y_trainval = X[: n_train + n_val], y[: n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

    # --- Evaluate k values on validation set ---
    val_auc_list = []
    for k in k_list:
        model = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)
        model.fit(X_train, y_train)

        y_val_pred = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred)
        val_auc_list.append(val_auc)

    # --- Select best k ---
    best_idx = int(np.argmax(val_auc_list))
    best_k = k_list[best_idx]

    # --- Retrain using train+val data ---
    best_model = KNeighborsClassifier(n_neighbors=best_k, metric=metric, n_jobs=-1)
    best_model.fit(X_trainval, y_trainval)

    # --- Evaluate on test set ---
    y_test_pred = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred)

    if return_auc_list:
        return test_auc, best_k, val_auc_list
    return test_auc, best_k


def knn_with_precomputed(
    dist_matrix: np.ndarray,
    y: np.ndarray,
    k_list: list[int],
    return_auc_list: bool = False,
    random_state: int = 42,
):
    """
    Perform k-Nearest Neighbors (kNN) classification using a precomputed distance matrix
    with an automatic 60/20/20 train/validation/test split.

    Parameters
    ----------
    dist_matrix : np.ndarray of shape (n_samples, n_samples)
        Square, precomputed distance matrix (zeros on diagonal).
        Rows represent query points, columns represent reference points.

    y : np.ndarray of shape (n_samples,)
        Binary class labels (0 or 1).

    k_list : list[int]
        List of k (neighbor counts) to evaluate during model selection.

    return_auc_list : bool, optional (default=False)
        If True, also returns the validation AUC scores for all k.

    random_state : int, optional (default=42)
        Random seed for reproducibility.

    Returns
    -------
    test_auc : float
        AUC score on the test set using the best-performing k.

    best_k : int
        The neighbor count (k) that achieved the highest validation AUC.

    val_auc_list : list[float], optional
        Validation AUCs for each k (only returned if `return_auc_list=True`).
    """

    # --- Shuffle dataset (Rows AND Columns) ---
    # When using a precomputed matrix, we must shuffle the indices identically
    # for both axes to preserve the relationship between sample i and sample j.
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(dist_matrix))

    # Shuffle matrix: reorder rows, then reorder columns
    dist_matrix = dist_matrix[indices][:, indices]
    y = y[indices]

    # --- Split sizes ---
    n_samples = len(dist_matrix)
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)
    # n_test is the remainder

    # --- Dataset splits (Labels) ---
    y_train = y[:n_train]
    y_val = y[n_train : n_train + n_val]
    y_trainval = y[: n_train + n_val]
    y_test = y[n_train + n_val :]

    # --- Dataset splits (Distance Matrices) ---
    # For metric='precomputed':
    # 1. Fit requires a square matrix (n_train x n_train)
    # 2. Predict requires a rectangular matrix (n_query x n_reference)

    # Training set (Square: Train vs Train)
    X_train = dist_matrix[:n_train, :n_train]

    # Validation set (Rectangular: Val rows vs Train cols)
    X_val = dist_matrix[n_train : n_train + n_val, :n_train]

    # Combined Train+Val set (Square: TrainVal vs TrainVal)
    X_trainval = dist_matrix[: n_train + n_val, : n_train + n_val]

    # Test set (Rectangular: Test rows vs TrainVal cols)
    # Note: We query Test points against the combined Train+Val set
    X_test = dist_matrix[n_train + n_val :, : n_train + n_val]

    # --- Evaluate k values on validation set ---
    val_auc_list = []
    for k in k_list:
        # Crucial: Set metric to 'precomputed'
        model = KNeighborsClassifier(n_neighbors=k, metric="precomputed", n_jobs=-1)
        model.fit(X_train, y_train)

        # X_val shape is (n_val_samples, n_train_samples)
        y_val_pred = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred)
        val_auc_list.append(val_auc)

    # --- Select best k ---
    best_idx = int(np.argmax(val_auc_list))
    best_k = k_list[best_idx]

    # --- Retrain using train+val data ---
    best_model = KNeighborsClassifier(
        n_neighbors=best_k, metric="precomputed", n_jobs=-1
    )
    # Fit on the larger square matrix
    best_model.fit(X_trainval, y_trainval)

    # --- Evaluate on test set ---
    # Predict using the distances from Test samples to TrainVal samples
    y_test_pred = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred)

    if return_auc_list:
        return test_auc, best_k, val_auc_list
    return test_auc, best_k


def slice_from_dist_matrix(dist_matrix, idx_list, copy=True):
    sliced_dist_matrix = dist_matrix[np.ix_(idx_list, idx_list)]

    if copy:
        sliced_dist_matrix = np.copy(sliced_dist_matrix)

    return sliced_dist_matrix


def fisher_ratio(self_matrix_1, self_matrix_2, off_diag_matrix):
    n_1 = self_matrix_1.shape[0]
    n_2 = self_matrix_2.shape[0]

    i_ut_1, j_ut_1 = np.triu_indices(n_1, k=1)
    i_ut_2, j_ut_2 = np.triu_indices(n_2, k=1)

    unique_self_distances_1 = self_matrix_1[i_ut_1, j_ut_1]
    unique_self_distances_2 = self_matrix_2[i_ut_2, j_ut_2]

    ratio = np.mean(off_diag_matrix) / (
        0.5 * (np.mean(unique_self_distances_1) + np.mean(unique_self_distances_2))
    )

    return ratio
