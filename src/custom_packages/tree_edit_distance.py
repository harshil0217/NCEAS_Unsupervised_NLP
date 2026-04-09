import networkx as nx


def anytree_to_networkx(root):
    """Convert an anytree Node hierarchy to a networkx DiGraph.

    Each node's ``name`` attribute becomes the node ID. Edges run
    parent → child.

    Parameters
    ----------
    root : anytree.Node
        Root of the anytree hierarchy.

    Returns
    -------
    nx.DiGraph
    """
    G = nx.DiGraph()
    stack = [root]
    while stack:
        node = stack.pop()
        G.add_node(node.name)
        for child in node.children:
            G.add_edge(node.name, child.name)
            stack.append(child)
    return G


class GreedyEditDistance:
    """Greedy approximation of Graph/Tree Edit Distance.

    All four edit costs default to 1 (node insertion, node deletion,
    edge insertion, edge deletion). Substitution is treated as a
    deletion + insertion (cost 2).

    Usage mirrors ged4py.GraphEditDistance::

        ged = GreedyEditDistance(1, 1, 1, 1)
        result = ged.compare([g1, g2], None)
        distance = result[0][1]

    Matching strategy
    -----------------
    1. Leaf nodes with the same ID are matched exactly (leaf IDs are
       shared data-point indices in this project's trees).
    2. Internal nodes are matched greedily by Jaccard similarity of
       their leaf-descendant sets, largest nodes first.
    3. Unmatched nodes incur insertion / deletion costs; after mapping,
       edges present in one graph but not the other incur edge costs.
    """

    def __init__(self, node_ins_cost=1, node_del_cost=1,
                 edge_ins_cost=1, edge_del_cost=1):
        self.node_ins = node_ins_cost
        self.node_del = node_del_cost
        self.edge_ins = edge_ins_cost
        self.edge_del = edge_del_cost

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _leaf_sets(self, G):
        """Return {node: frozenset of leaf descendants} for every node."""
        leaves = {n for n in G.nodes() if G.out_degree(n) == 0}
        leaf_sets = {}
        for node in reversed(list(nx.topological_sort(G))):
            if node in leaves:
                leaf_sets[node] = frozenset([node])
            else:
                combined = set()
                for child in G.successors(node):
                    combined |= leaf_sets.get(child, set())
                leaf_sets[node] = frozenset(combined)
        return leaf_sets

    def _greedy_match(self, g1, g2):
        """Return a node mapping g1_node -> g2_node via greedy matching."""
        ls1 = self._leaf_sets(g1)
        ls2 = self._leaf_sets(g2)

        leaves1 = {n for n in g1.nodes() if g1.out_degree(n) == 0}
        leaves2 = {n for n in g2.nodes() if g2.out_degree(n) == 0}

        node_map = {}   # g1_node -> g2_node
        matched1 = set()
        matched2 = set()

        # Step 1: match leaf nodes with identical IDs
        for leaf in leaves1 & leaves2:
            node_map[leaf] = leaf
            matched1.add(leaf)
            matched2.add(leaf)

        # Step 2: greedily match internal nodes by Jaccard of leaf sets
        internals1 = sorted(
            [n for n in g1.nodes() if n not in leaves1],
            key=lambda n: -len(ls1.get(n, set()))
        )
        unmatched_int2 = {n for n in g2.nodes() if n not in leaves2}

        for n1 in internals1:
            ls_n1 = ls1.get(n1, frozenset())
            if not ls_n1:
                continue

            best_j = 0.0
            best_n2 = None
            for n2 in unmatched_int2:
                ls_n2 = ls2.get(n2, frozenset())
                union = len(ls_n1 | ls_n2)
                if union == 0:
                    continue
                j = len(ls_n1 & ls_n2) / union
                if j > best_j:
                    best_j = j
                    best_n2 = n2

            if best_n2 is not None and best_j > 0.0:
                node_map[n1] = best_n2
                matched1.add(n1)
                matched2.add(best_n2)
                unmatched_int2.remove(best_n2)

        return node_map, matched1, matched2

    def _ged(self, g1, g2):
        node_map, matched1, matched2 = self._greedy_match(g1, g2)
        reverse_map = {v: k for k, v in node_map.items()}

        cost = 0

        # Unmatched node costs
        cost += len(set(g1.nodes()) - matched1) * self.node_del
        cost += len(set(g2.nodes()) - matched2) * self.node_ins

        # Edge costs: edges in g1 not covered in g2 after mapping
        for u, v in g1.edges():
            mu, mv = node_map.get(u), node_map.get(v)
            if mu is None or mv is None or not g2.has_edge(mu, mv):
                cost += self.edge_del

        # Edge costs: edges in g2 not covered in g1 after reverse mapping
        for u, v in g2.edges():
            ou, ov = reverse_map.get(u), reverse_map.get(v)
            if ou is None or ov is None or not g1.has_edge(ou, ov):
                cost += self.edge_ins

        return cost

    # ------------------------------------------------------------------
    # Public API (mirrors ged4py.GraphEditDistance)
    # ------------------------------------------------------------------

    def compare(self, graphs, _):
        """Compute pairwise GED for a list of graphs.

        Parameters
        ----------
        graphs : list of nx.Graph / nx.DiGraph
        _ : ignored (present for API compatibility)

        Returns
        -------
        list of list of float
            result[i][j] is the greedy GED between graphs[i] and graphs[j].
        """
        n = len(graphs)
        result = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = float(self._ged(graphs[i], graphs[j]))
                result[i][j] = d
                result[j][i] = d
        return result
