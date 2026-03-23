import numpy as np

def build_dc_tree(history):
    n = len(history[0])
    parent = {}
    current_node_id = n
    cluster_nodes = {}

    for i in range(n):
        cluster_nodes[(0, history[0][i], i)] = i

    for t in range(len(history) - 1):
        labels_t = history[t]
        labels_next = history[t + 1]

        for c in np.unique(labels_next):
            idx = np.where(labels_next == c)[0]
            prev_clusters = set(labels_t[i] for i in idx)

            if len(prev_clusters) > 1:
                new_node = current_node_id
                current_node_id += 1

                for i in idx:
                    child = cluster_nodes.get((t, labels_t[i], i), i)
                    parent[child] = new_node

                for i in idx:
                    cluster_nodes[(t + 1, c, i)] = new_node
            else:
                pc = list(prev_clusters)[0]
                for i in idx:
                    cluster_nodes[(t + 1, c, i)] = cluster_nodes.get((t, pc, i), i)

    return parent


def get_ancestors(node, parent):
    ancestors = set()
    while node in parent:
        node = parent[node]
        ancestors.add(node)
    return ancestors


def compute_lca(i, j, parent):
    ancestors_i = get_ancestors(i, parent)
    while j in parent:
        j = parent[j]
        if j in ancestors_i:
            return j
    return None


def lca_f1_dc(parent, true_labels, max_pairs=50000):
    n = len(true_labels)
    rng = np.random.default_rng(42)

    correct = 0
    for _ in range(max_pairs):
        i, j = rng.choice(n, 2, replace=False)

        same_true = (true_labels[i] == true_labels[j])
        same_pred = compute_lca(i, j, parent) is not None

        if same_true == same_pred:
            correct += 1

    return correct / max_pairs


def tree_edit_distance_dc(parent, true_labels, max_pairs=50000):
    n = len(true_labels)
    rng = np.random.default_rng(42)

    mismatches = 0
    for _ in range(max_pairs):
        i, j = rng.choice(n, 2, replace=False)

        same_true = (true_labels[i] == true_labels[j])
        same_pred = compute_lca(i, j, parent) is not None

        if same_true != same_pred:
            mismatches += 1

    return mismatches / max_pairs