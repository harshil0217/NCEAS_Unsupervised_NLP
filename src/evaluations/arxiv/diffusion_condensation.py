import numpy as np
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.preprocessing import normalize

class DiffusionCondensation:
    def __init__(
        self,
        k=10,
        alpha=2,
        merge_threshold=1e-3,
        min_clusters=5,
        max_iterations=10,   # 🔥 reduced
        random_state=42
    ):
        self.k = k
        self.alpha = alpha
        self.merge_threshold = merge_threshold
        self.min_clusters = min_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state

        self.labels_ = None
        self.history_ = []

    def _build_graph(self, X):
        A = kneighbors_graph(X, self.k, mode="connectivity", include_self=True)
        return A.toarray()

    def _diffuse(self, X, A):
        A = normalize(A, norm="l1", axis=1)
        return A @ X

    # ✅ FAST MERGE (FIXED)
    def _merge(self, X):
        n = len(X)
        labels = -np.ones(n, dtype=int)

        nbrs = NearestNeighbors(n_neighbors=5, n_jobs=1)
        nbrs.fit(X)

        cluster_id = 0

        for i in range(n):
            if labels[i] != -1:
                continue

            neighbors = nbrs.kneighbors([X[i]], return_distance=False)[0]

            for j in neighbors:
                labels[j] = cluster_id

            cluster_id += 1

        return labels

    def fit(self, X):
        np.random.seed(self.random_state)

        X_current = X.copy()
        A = self._build_graph(X_current)

        self.history_ = []

        for it in range(self.max_iterations):
            print(f"Iteration {it+1}/{self.max_iterations}")

            X_current = self._diffuse(X_current, A)
            labels = self._merge(X_current)

            self.labels_ = labels
            self.history_.append(labels.copy())

            num_clusters = len(np.unique(labels))
            print(f"Clusters: {num_clusters}")

            if num_clusters <= self.min_clusters:
                print("Stopping early")
                break

        return self