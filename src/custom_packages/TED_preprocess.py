"""
TED_preprocess.py

Converts predicted and ground-truth trees to the children-list format required
by the APTED / TED algorithm, writes both representations stacked vertically to
a txt file, and writes an identical copy alongside it.

Format (one line per tree):
    [[child_ids_of_node_0], [child_ids_of_node_1], [], ...]

Nodes are indexed by their anytree .name attribute; gaps between IDs become
empty lists. The pre-order traversal is performed by anytree_to_children_list
(imported from lca_f1.py).

Usage
-----
    from custom_packages.TED_preprocess import write_ted_input

    write_ted_input(pred_tree, gt_tree, "path/to/output.txt")
    # Produces output.txt and output_copy.txt
"""

import os
import shutil
from scipy.cluster.hierarchy import ClusterNode
from anytree import Node

from custom_packages.lca_f1 import clusternode_to_anytree, anytree_to_children_list


def _to_anytree(tree):
    """Ensure the tree is an anytree Node, converting from ClusterNode if needed."""
    if isinstance(tree, ClusterNode):
        return clusternode_to_anytree(tree)
    if isinstance(tree, Node):
        return tree
    raise TypeError(f"Unsupported tree type: {type(tree)}. Expected ClusterNode or anytree.Node.")


def _children_list_to_str(children_list):
    """Render a children-list as the compact bracket string used by APTED."""
    return "[" + ", ".join(str(children) for children in children_list) + "]"


def write_ted_input(pred_tree, gt_tree, output_path):
    """
    Write predicted and ground-truth trees to a TED-compatible txt file and
    produce an identical copy.

    Parameters
    ----------
    pred_tree : ClusterNode or anytree.Node
        Predicted hierarchy (root node).
    gt_tree : ClusterNode or anytree.Node
        Ground-truth hierarchy (root node).
    output_path : str
        Path for the primary output file (e.g. "ted_input.txt").
        The copy is written to the same directory with "_copy" appended before
        the extension (e.g. "ted_input_copy.txt").

    Returns
    -------
    copy_path : str
        Path of the copy file that was written.
    """
    pred_root = _to_anytree(pred_tree)
    gt_root = _to_anytree(gt_tree)

    pred_children = anytree_to_children_list(pred_root)
    gt_children = anytree_to_children_list(gt_root)

    pred_line = _children_list_to_str(pred_children)
    gt_line = _children_list_to_str(gt_children)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(pred_line + "\n")
        f.write(gt_line + "\n")

    base, ext = os.path.splitext(output_path)
    copy_path = base + "_copy" + ext
    shutil.copy2(output_path, copy_path)

    print(f"TED input written to:      {output_path}")
    print(f"Copy written to:           {copy_path}")

    return copy_path
