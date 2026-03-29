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

class UnionFind:
    def __init__(self, n):
        self.parent = -1 * np.ones(2 * n - 1, dtype=np.intp)
        self.size = np.hstack([np.ones(n, dtype=np.intp), np.zeros(n - 1, dtype=np.intp)])
        self.next_label = n

    def union(self, a, b):
        self.size[self.next_label] = self.size[a] + self.size[b]
        self.parent[a] = self.next_label
        self.parent[b] = self.next_label
        self.next_label += 1

    def find(self, n):
        p = n
        while self.parent[n] != -1:
            n = self.parent[n]
        while self.parent[p] != n:
            p, self.parent[p] = self.parent[p], n
        return n

    
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

    def merge_data_points(self, data):
        """
        Merges data points that are within the merge_threshold distance of each other.
        Keeps track of which points are merged into which.
        """
        data_gpu = cupy.asarray(data)
        distance_matrix = cupy.asnumpy(pairwise_distances(data_gpu, metric='euclidean'))
        numpoints = distance_matrix.shape[0]
        cluster_mapping = {i: i for i in range(numpoints)}
        indices_to_keep = list(set(range(numpoints)))
        new_merges = False
        linkage_rows = []                                       

        # Iterate over data labels starting with highest number to push labels to lower values
        for j in np.arange(numpoints - 1, 0, -1):
            min_dist = np.min(distance_matrix[j, :j])
            if min_dist < self.merge_threshold:
                new_merges = True
                # Merge with closest index
                target = np.argmin(distance_matrix[j, :j])
                cluster_mapping[j] = target

                a = self.uf.find(self.translation[j])
                b = self.uf.find(self.translation[target])       
                if a != b:                                         
                    linkage_rows.append([a, b, min_dist,          
                        self.uf.size[a] + self.uf.size[b]])       
                    self.uf.union(a, b)                            

                # Discard datapoint in future condensation iterations
                indices_to_keep.remove(j)
                for i in np.arange(numpoints - 1, j, -1):
                    # Relabel clusters whose label is the current relabeled node
                    if cluster_mapping[i] == j:
                        cluster_mapping[i] = cluster_mapping[j]
                    # Shift higher indices to lower value (for correct future indexing)
                    elif cluster_mapping[i] > j:
                        cluster_mapping[i] -= 1

         # NEW — rebuild translation for compacted indices
        self.translation = {new_idx: self.translation[old_idx]
                        for new_idx, old_idx in enumerate(indices_to_keep)}
        
        new_data = data[indices_to_keep]

        if new_merges:
            return new_data, cluster_mapping, linkage_rows
        else:
            # No need to pass cluster mapping if no merges occur
            return new_data, None, []


    def fit(self, data, prev_cluster_tree=None, prev_data=None):
        n = data.shape[0]
        self.translation = {i: i for i in range(n)}               
        self.uf = UnionFind(n)                                    
        all_linkage_rows = []                                      


        if prev_cluster_tree is not None and prev_data is not None:
            self.cluster_tree = prev_cluster_tree
            data = prev_data
            num_clusters = data.shape[0]
            iterations = len(prev_cluster_tree) - 1
        else:
            num_clusters = data.shape[0]
            self.cluster_tree = [{
                "clusters": num_clusters,
                "cluster_maps": {i: i for i in range(num_clusters)}
            }]
            iterations = 0

        while iterations < self.max_iterations and num_clusters > self.min_clusters:
            self.k = int(self.interpolate_param(self.k, self.k_end, iterations, self.max_iterations))
            self.t = int(self.interpolate_param(self.t, self.t_end, iterations, self.max_iterations))
            self.alpha = self.interpolate_param(self.alpha, self.alpha_end, iterations, self.max_iterations)
            self.merge_threshold = self.interpolate_param(self.merge_threshold, self.merge_threshold_end, iterations, self.max_iterations)
            
            data = self.diffusion_condensation(data)
            data, mapping, linkage_rows = self.merge_data_points(data)
            all_linkage_rows.extend(linkage_rows)
            if mapping is not None:
                self.cluster_tree.append({
                    "clusters": data.shape[0],
                    "cluster_maps": mapping
                })

            num_clusters = data.shape[0]
            iterations += 1

        Z = np.array(all_linkage_rows, dtype=float)
        # topological sort — a row can only appear after both its children are defined
        order = []
        available = set(range(n))
        rows_remaining = list(enumerate(Z))

        while rows_remaining:
            for idx, row in rows_remaining:
                a, b = int(row[0]), int(row[1])
                if a in available and b in available:
                    order.append(idx)
                    available.add(n + len(order) - 1)
                    rows_remaining.remove((idx, row))
                    break

        Z = Z[order]
        self.linkage_matrix_ = Z
            
    def interpolate_param(self, start, end, iteration, max_iterations):
        if end is None:
            return start
        # Ease-in function: fast early movement, then slow convergence
        progress = iteration / max_iterations
        eased_progress = 1 - np.exp(-5 * progress)  # The constant (5) controls steepness
        return start + (end - start) * eased_progress
    # def interpolate_param(self, start, end, iteration, max_iterations):
    #     if end is None:
    #         return start
    #     return start + (end - start) * (iteration / max_iterations)

    def get_labels(self, cluster_level=None):
        """Retrieves coarsest cluster ids for datapoints"""
        labels = self.cluster_tree[0]["cluster_maps"]
        for map in self.cluster_tree[1:cluster_level]:
            temp = {}
            for datapoint_id, label in labels.items():
                temp[datapoint_id] = map["cluster_maps"][label]

            labels = temp

        self.labels_ = np.array([v for v in labels.values()])

    def predict(self):
        return None