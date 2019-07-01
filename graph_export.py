"""
Helper functions to export trees as pdf
"""
from sklearn.tree import export_graphviz
import pydotplus


def export_tree(tree, file_name, feature_names=None):
    if tree is not None:
        dot_data = export_graphviz(tree, feature_names=feature_names, out_file=None, filled=True, rounded=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph = make_graph_minimal(graph)

        graph.write_pdf(file_name)


def make_graph_minimal(graph):
    nodes = graph.get_nodes()
    for node in nodes:
        old_label = node.get_label()
        label = prune_label(old_label)
        if label is not None:
            node.set_label(label)
    return graph


def prune_label(label):
    if label is None:
        return None
    if len(label) == 0:
        return None
    label = label[1:-1]
    parts = [part for part in label.split('\\n')
             if 'gini =' not in part and 'samples =' not in part]
    return '"' + '\\n'.join(parts) + '"'