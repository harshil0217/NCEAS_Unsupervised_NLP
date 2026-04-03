import numpy as np
from scipy.cluster.hierarchy import ClusterNode
from anytree import Node


def clusternode_to_anytree(cluster_node):
    """
    Convert scipy ClusterNode (binary tree) to anytree Node (multi-way tree).
    Uses iterative approach to avoid recursion depth issues.

    Parameters
    ----------
    cluster_node : ClusterNode
        Root of the scipy ClusterNode tree

    Returns
    -------
    Node
        Root of the anytree Node tree
    """
    root = Node(name=cluster_node.id)
    stack = [(cluster_node, root)]

    while stack:
        cn, an_parent = stack.pop()

        if not cn.is_leaf():
            if cn.left is not None:
                left_node = Node(name=cn.left.id, parent=an_parent)
                stack.append((cn.left, left_node))
            if cn.right is not None:
                right_node = Node(name=cn.right.id, parent=an_parent)
                stack.append((cn.right, right_node))

    return root


def get_leaves(node):
    """Get all leaf node ids under a node using iterative DFS."""
    leaves = []
    stack = [node]
    while stack:
        current = stack.pop()
        if current.is_leaf:
            leaves.append(current.name)
        else:
            for child in current.children:
                stack.append(child)
    return leaves


def build_maps(root):
    """Build node map and parent map in a single traversal."""
    node_map = {}
    parent_map = {root.name: None}
    stack = [root]
    while stack:
        node = stack.pop()
        node_map[node.name] = node
        for child in node.children:
            parent_map[child.name] = node.name
            stack.append(child)
    return node_map, parent_map


def get_ancestors(node_id, parent_map):
    """Get list of ancestors from node to root (inclusive) using parent pointers."""
    ancestors = []
    current = node_id
    while current is not None:
        ancestors.append(current)
        current = parent_map[current]
    return ancestors


def find_lca(i, j, parent_map):
    """Find LCA by comparing ancestor sets."""
    ancestors_i = get_ancestors(i, parent_map)
    ancestors_j_set = set(get_ancestors(j, parent_map))

    for ancestor in ancestors_i:
        if ancestor in ancestors_j_set:
            return ancestor
    return None


def lca_f1(pred_tree, gt_tree, true_labels, n_samples=1000, n_trials=5):
    """
    Monte Carlo estimation of LCA-F1 for hierarchical clustering evaluation.

    For each sample: picks a ground-truth cluster proportional to its number of pairs,
    selects two random points from it, finds their LCA in both the predicted and ground
    truth trees, then computes per-sample F1 from the overlap of the two LCA subtrees.
    Runs n_trials independent trials of n_samples each and returns the mean.

    Parameters
    ----------
    pred_tree : ClusterNode or anytree.Node
        Root of the predicted hierarchy.
    gt_tree : ClusterNode or anytree.Node
        Root of the ground truth hierarchy.
    true_labels : array-like of shape (n_samples,)
        Ground-truth flat cluster labels for each data point (index = leaf node id).
    n_samples : int, optional
        Number of Monte Carlo samples per trial (default 10_000).
    n_trials : int, optional
        Number of independent trials to average over (default 30).

    Returns
    -------
    float
        Mean LCA-F1 score across all trials (0–1).
    """
    if isinstance(pred_tree, ClusterNode):
        pred_tree = clusternode_to_anytree(pred_tree)
    if isinstance(gt_tree, ClusterNode):
        gt_tree = clusternode_to_anytree(gt_tree)

    true_labels = np.asarray(true_labels)

    pred_node_map, pred_parent_map = build_maps(pred_tree)
    gt_node_map, gt_parent_map = build_maps(gt_tree)

    true_clusters = np.unique(true_labels)
    cluster_counts = np.array([np.sum(true_labels == c) for c in true_clusters])

    # pair-weighted sampling: larger clusters contribute more pairs
    sort_idx = np.argsort(cluster_counts)[::-1]
    true_clusters_sorted = true_clusters[sort_idx]
    cluster_counts_sorted = cluster_counts[sort_idx]

    weights = cluster_counts_sorted * (cluster_counts_sorted - 1) / 2  # C(k,2) pairs
    weights = weights / weights.sum()

    trial_means = []

    for _ in range(n_trials):
        f1_scores = []

        for _ in range(n_samples):
            c = np.random.choice(true_clusters_sorted, p=weights)
            cluster_idx = np.where(true_labels == c)[0]
            i, j = np.random.choice(cluster_idx, size=2, replace=False)

            pred_lca_id = find_lca(i, j, pred_parent_map)
            gt_lca_id = find_lca(i, j, gt_parent_map)

            pred_leaves = set(get_leaves(pred_node_map[pred_lca_id]))
            gt_leaves = set(get_leaves(gt_node_map[gt_lca_id]))

            intersection = len(pred_leaves & gt_leaves)
            precision = intersection / len(pred_leaves)
            recall = intersection / len(gt_leaves)

            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * precision * recall / (precision + recall))

        trial_means.append(np.mean(f1_scores))

    return float(np.mean(trial_means))