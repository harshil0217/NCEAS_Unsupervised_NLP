import numpy as np
from scipy.cluster.hierarchy import to_tree

def dendrogram_purity(tree, true_labels):
    def get_leaves(node):
        if node.is_leaf():
            return [node.id]
        return get_leaves(node.left) + get_leaves(node.right)

    def build_node_map(root, node_map=None):
        """Build a mapping from node id to node object."""
        if node_map is None:
            node_map = {}
        node_map[root.id] = root
        if not root.is_leaf():
            build_node_map(root.left, node_map)
            build_node_map(root.right, node_map)
        return node_map

    def get_root_to_leaf_path(root, target_id):
        """Returns list of node ids from root down to the target leaf."""
        if root is None:
            return None
        if root.id == target_id:
            return [root.id]

        # Try left subtree
        if root.left is not None:
            left_path = get_root_to_leaf_path(root.left, target_id)
            if left_path is not None:
                return [root.id] + left_path

        # Try right subtree
        if root.right is not None:
            right_path = get_root_to_leaf_path(root.right, target_id)
            if right_path is not None:
                return [root.id] + right_path

        return None

    def find_lca(root, i, j):
        path_i = get_root_to_leaf_path(root, i)
        path_j = get_root_to_leaf_path(root, j)

        # LCA is the last common node in both paths
        lca_id = None
        for a, b in zip(path_i, path_j):
            if a == b:
                lca_id = a
            else:
                break

        return lca_id  # returns the node id

    # Build node id to node object mapping
    node_map = build_node_map(tree)

    # get true cluster assignments
    true_clusters = np.unique(true_labels)

    N_SAMPLES = 10_000

    scores = []
    cluster_counts = np.array([np.sum(true_labels == c) for c in true_clusters])

    # sort clusters by size descending and compute pair-weighted sampling probs
    # larger clusters are sampled more often since they contain more pairs
    sort_idx = np.argsort(cluster_counts)[::-1]
    true_clusters_sorted = true_clusters[sort_idx]
    cluster_counts_sorted = cluster_counts[sort_idx]

    weights = cluster_counts_sorted * (cluster_counts_sorted - 1) / 2  # C(k,2) pairs per cluster
    weights = weights / weights.sum()  # normalize to valid probability distribution

    for _ in range(N_SAMPLES):
        # sample a cluster proportional to its number of pairs, then pick 2 points from it
        c = np.random.choice(true_clusters_sorted, p=weights)
        cluster_idx = np.where(true_labels == c)[0]
        i, j = np.random.choice(cluster_idx, size=2, replace=False)

        # find lowest common ancestor of i and j in the dendrogram
        lca_id = find_lca(tree, i, j)
        lca_node = node_map[lca_id]
        lca_leaves = get_leaves(lca_node)

        # purity = fraction of LCA's subtree that belongs to the same true cluster
        purity = np.sum(true_labels[lca_leaves] == c) / len(lca_leaves)
        scores.append(purity)

    # average purity over all sampled pairs = dendrogram purity estimate
    dendrogram_purity = np.mean(scores)

    return dendrogram_purity



    
