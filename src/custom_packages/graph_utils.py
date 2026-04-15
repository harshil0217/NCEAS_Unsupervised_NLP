"""
graph_utils.py

Tree utility functions shared across evaluation metrics. Handles conversion
between scipy ClusterNode dendrograms and anytree Node trees, leaf extraction,
LCA (lowest common ancestor) lookup, and APTED tree edit distance computation.
"""

import networkx as nx
from scipy.cluster.hierarchy import ClusterNode
from anytree import Node
from apted import APTED, Config


class AnyTreeAPTEDConfig(Config):
    """APTED config for anytree.Node trees.

    Rename cost is always 0 — predicted tree internal node names are arbitrary,
    so only structural differences (insertions/deletions, cost 1 each) are counted.
    """
    def rename(self, node1, node2):
        return 0

    def children(self, node):
        return list(node.children)


def apted_distance(tree1, tree2):
    """Compute APTED tree edit distance between two anytree.Node trees."""
    return APTED(tree1, tree2, AnyTreeAPTEDConfig()).compute_edit_distance()


def anytree_to_networkx(root):
    """Convert an anytree Node hierarchy to a networkx DiGraph.

    Each node's ``name`` attribute becomes the node ID. Edges run
    child → parent.
    """
    G = nx.DiGraph()
    stack = [root]
    while stack:
        node = stack.pop()
        G.add_node(node.name)
        for child in node.children:
            G.add_edge(child.name, node.name)
            stack.append(child)
    return G


def anytree_to_zss(root):
    """Convert an anytree Node hierarchy to a zss.Node tree for ZSS tree edit distance.

    Node labels are the string representation of each node's ``name`` attribute.
    """
    import zss
    zss_root = zss.Node(str(root.name))
    stack = [(root, zss_root)]
    while stack:
        an_node, zss_node = stack.pop()
        for child in an_node.children:
            zss_child = zss.Node(str(child.name))
            zss_node.addkid(zss_child)
            stack.append((child, zss_child))
    return zss_root


def clusternode_to_anytree(cluster_node):
    """Convert scipy ClusterNode (binary tree) to anytree Node (multi-way tree).

    Uses iterative approach to avoid recursion depth issues.
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


def anytree_to_children_list(root):
    """Convert anytree to a 2D children-list using pre-order traversal.

    Returns a list indexed by node ID where each entry is the list of
    that node's children IDs. Indices between 0 and max node ID with no
    node will be empty lists.
    """
    max_id = 0
    stack = [root]
    while stack:
        node = stack.pop()
        if node.name > max_id:
            max_id = node.name
        for child in node.children:
            stack.append(child)

    result = [[] for _ in range(max_id + 1)]

    stack = [root]
    while stack:
        node = stack.pop()
        result[node.name] = [child.name for child in node.children]
        for child in reversed(node.children):
            stack.append(child)

    return result


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
