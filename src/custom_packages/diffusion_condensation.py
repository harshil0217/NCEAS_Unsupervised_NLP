from sklearn.preprocessing import normalize
import sklearn
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
import scipy
from cuml.neighbors import NearestNeighbors
from cuml.metrics import pairwise_distances
import cupy
import cupyx


class DiffusionCondensation:
    def __init__(self,
                 k=5,
                 alpha=2,
                 epsilon_scale=0.99,
                 merge_threshold=1e-15,
                 merge_threshold_end=None,  # NEW
                 min_clusters=5,
                 max_iterations=1000,
                 t=1,
                 data_dependent_epsilon=True,
                 symmetric_kernel=False,
                 bandwidth_norm="max",
                 k_end=None,
                 t_end=None,
                 alpha_end=None):

        self.k = k
        self.k_end = k_end
        self.alpha = alpha
        self.alpha_end = alpha_end
        self.epsilon_scale = epsilon_scale
        self.merge_threshold = merge_threshold
        self.merge_threshold_end = merge_threshold_end  # NEW
        self.min_clusters = min_clusters
        self.max_iterations = max_iterations
        self.t = t
        self.t_end = t_end
        self.data_dependent_epsilon = data_dependent_epsilon
        self.symmetric_kernel = symmetric_kernel
        self.bandwidth_norm = bandwidth_norm
        self.cluster_function = None
        self.epsilon = None
        self.labels_ = None
        self.cluster_tree = None


    def diffusion_operator(self, data):
        """Create diffusion operator"""
        # Generate the k-NN graph adjacency matrix using cuML
        knn = NearestNeighbors(n_neighbors = self.k)
        knn.fit(data)
        distances, indices = knn.kneighbors(data)

        distances = cupy.asnumpy(distances).reshape(data.shape[0] * self.k)
        indices = cupy.asnumpy(indices).reshape(data.shape[0] * self.k)
        indptr = np.arange(0, (self.k * data.shape[0]) + 1, self.k)

        knn_graph = scipy.sparse.csr_matrix(
            (distances, indices, indptr),
            shape=(data.shape[0], data.shape[0])
        )
        knn_graph_adjacency = knn_graph.maximum(knn_graph.T)

        knn_graph_adjacency = normalize(knn_graph_adjacency, norm=self.bandwidth_norm, axis=1, copy=False)

        knn_graph_adjacency = -knn_graph_adjacency.power(self.alpha)
        np.exp(knn_graph_adjacency.data, out=knn_graph_adjacency.data)

        if self.symmetric_kernel:
            knn_graph_adjacency = 0.5*(knn_graph_adjacency + knn_graph_adjacency.T)
        knn_graph_adjacency = normalize(knn_graph_adjacency, norm='l1', axis=1, copy=False)

        return knn_graph_adjacency

    def diffusion_condensation(self, data):
        """Run Diffusion Condensation"""
        if self.k >= data.shape[0]:
            self.k = data.shape[0] - 1

        P = self.diffusion_operator(data)
        for _ in np.arange(self.t):
            data = P @ data

        return data

    def merge_data_points(self, data, row_to_cluster, cluster_sizes, next_cluster_id):
        """
        Merges data points that are within the merge_threshold distance of each other.
        Creates new cluster indices for merged clusters instead of relabeling.

        Args:
            data: current data array
            row_to_cluster: list mapping each row index to its cluster ID
            cluster_sizes: dict mapping cluster ID to number of original points
            next_cluster_id: next available cluster ID for merged clusters

        Returns:
            new_data: condensed data array
            new_row_to_cluster: updated mapping from row index to cluster ID
            cluster_sizes: updated cluster sizes dict
            next_cluster_id: updated next cluster ID
            merges: list of (cluster_a, cluster_b, new_cluster_id, distance, size) tuples
        """
        data_gpu = cupy.asarray(data)
        distance_matrix = cupy.asnumpy(pairwise_distances(data_gpu, metric='euclidean'))
        numpoints = distance_matrix.shape[0]

        merged = set()  # indices of rows that have been merged into another
        merges = []  # merge events: (cluster_a, cluster_b, new_cluster_id, distance, size)

        # Process from highest index to lowest
        for j in range(numpoints - 1, 0, -1):
            if j in merged:
                continue

            # Find closest point with index < j that hasn't been merged
            min_dist = float('inf')
            target = -1
            for i in range(j):
                if i not in merged and distance_matrix[j, i] < min_dist:
                    min_dist = distance_matrix[j, i]
                    target = i

            if target >= 0 and min_dist < self.merge_threshold:
                # Merge row j into row target
                cluster_j = row_to_cluster[j]
                cluster_target = row_to_cluster[target]

                new_size = cluster_sizes[cluster_j] + cluster_sizes[cluster_target]

                # Record merge: clusters cluster_j and cluster_target merge into next_cluster_id
                merges.append((cluster_j, cluster_target, next_cluster_id, min_dist, new_size))

                # Update: target row now represents the new merged cluster
                row_to_cluster[target] = next_cluster_id
                cluster_sizes[next_cluster_id] = new_size

                # Mark row j as merged
                merged.add(j)

                next_cluster_id += 1

        # Build new data array and row_to_cluster for kept rows
        indices_to_keep = [i for i in range(numpoints) if i not in merged]
        new_data = data[indices_to_keep]
        new_row_to_cluster = [row_to_cluster[i] for i in indices_to_keep]

        if merges:
            return new_data, new_row_to_cluster, cluster_sizes, next_cluster_id, merges
        else:
            return new_data, new_row_to_cluster, cluster_sizes, next_cluster_id, None


    def fit(self, data, prev_cluster_tree=None, prev_data=None):
        n = data.shape[0]

        if prev_cluster_tree is not None and prev_data is not None:
            self.cluster_tree = prev_cluster_tree
            data = prev_data
            num_clusters = data.shape[0]
            iterations = len(prev_cluster_tree)
            # Reconstruct state from previous cluster_tree
            next_cluster_id = n + len(prev_cluster_tree)
            # Rebuild cluster_sizes from merges
            cluster_sizes = {i: 1 for i in range(n)}
            for cluster_a, cluster_b, new_id, dist, size in prev_cluster_tree:
                cluster_sizes[new_id] = size
            # row_to_cluster needs to be reconstructed - for simplicity, derive from remaining clusters
            # This is an approximation; full reconstruction would require tracking active clusters
            row_to_cluster = list(range(num_clusters))
        else:
            num_clusters = data.shape[0]
            # row_to_cluster[i] = cluster ID that data row i represents
            # Initially, each row represents its own cluster (leaf node)
            row_to_cluster = list(range(n))
            # cluster_sizes tracks number of original points in each cluster
            cluster_sizes = {i: 1 for i in range(n)}
            # next_cluster_id for merged clusters (starts at n, leaf nodes are 0 to n-1)
            next_cluster_id = n
            # cluster_tree stores merge events: (cluster_a, cluster_b, new_cluster_id, distance, size)
            self.cluster_tree = []
            iterations = 0

        while iterations < self.max_iterations and num_clusters > self.min_clusters:
            self.k = int(self.interpolate_param(self.k, self.k_end, iterations, self.max_iterations))
            self.t = int(self.interpolate_param(self.t, self.t_end, iterations, self.max_iterations))
            self.alpha = self.interpolate_param(self.alpha, self.alpha_end, iterations, self.max_iterations)
            self.merge_threshold = self.interpolate_param(self.merge_threshold, self.merge_threshold_end, iterations, self.max_iterations)

            data = self.diffusion_condensation(data)
            data, row_to_cluster, cluster_sizes, next_cluster_id, merges = self.merge_data_points(
                data, row_to_cluster, cluster_sizes, next_cluster_id
            )

            if merges is not None:
                self.cluster_tree.extend(merges)

            num_clusters = data.shape[0]
            iterations += 1

        # Build linkage matrix by iterating backwards through cluster_tree
        self._build_linkage_matrix(n)
            
    def interpolate_param(self, start, end, iteration, max_iterations):
        if end is None:
            return start
        # Ease-in function: fast early movement, then slow convergence
        progress = iteration / max_iterations
        eased_progress = 1 - np.exp(-5 * progress)  # The constant (5) controls steepness
        return start + (end - start) * eased_progress

    def _build_linkage_matrix(self, n):
        """
        Build scipy-compatible linkage matrix by iterating backwards through cluster_tree.

        The cluster_tree stores merge events as tuples:
            (cluster_a, cluster_b, new_cluster_id, distance, size)

        Leaf nodes are indices 0 to n-1 (preserved original indices).
        Merged clusters get indices n, n+1, n+2, ... in order of creation.

        Args:
            n: number of original data points (leaf nodes)
        """
        if not self.cluster_tree:
            self.linkage_matrix_ = np.array([]).reshape(0, 4)
            return

        # Iterate backwards through cluster_tree from end to beginning
        # and collect merge information in reverse order
        Z = []
        for i in range(len(self.cluster_tree) - 1, -1, -1):
            cluster_a, cluster_b, new_id, dist, size = self.cluster_tree[i]
            Z.append([cluster_a, cluster_b, dist, size])

        # Reverse to restore chronological order (cluster n first, then n+1, etc.)
        # This ensures Z[i] corresponds to creating cluster n+i
        Z.reverse()

        self.linkage_matrix_ = np.array(Z, dtype=float)

    def get_labels(self, n_clusters=None):
        """
        Retrieves cluster labels for all original data points at a given granularity.

        Args:
            n_clusters: number of clusters desired (if None, uses all merges)

        Returns:
            Sets self.labels_ to array of cluster assignments for each original point
        """
        if not self.cluster_tree:
            # No merges occurred, each point is its own cluster
            self.labels_ = np.arange(len(self.labels_) if self.labels_ is not None else 0)
            return

        # Determine how many merges to apply based on desired n_clusters
        n = int(self.linkage_matrix_[-1, 3]) if len(self.linkage_matrix_) > 0 else 0
        if n == 0:
            return

        # Use scipy's fcluster to cut the dendrogram at the appropriate level
        from scipy.cluster.hierarchy import fcluster

        if n_clusters is not None:
            self.labels_ = fcluster(self.linkage_matrix_, n_clusters, criterion='maxclust')
        else:
            # Return labels at the coarsest level (all points in one cluster)
            self.labels_ = np.zeros(n, dtype=int)

    def predict(self):
        return None