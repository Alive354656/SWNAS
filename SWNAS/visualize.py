from collections import namedtuple
from graphviz import Digraph


def plot(edges, nodes, file_path, caption=None):
    """ make DAG plot and save to file_path as .png """
    edge_attr = {
        'fontsize': '22',
        'fontname': 'times',
        'penwidth': '2'
    }
    node_attr = {
        'style': 'filled',
        'shape': 'rect',
        'align': 'center',
        'fontsize': '20',
        'height': '0.5',
        'width': '0.5',
        'penwidth': '2',
        'fontname': 'times'
    }
    g = Digraph(
        format='png',
        edge_attr=edge_attr,
        node_attr=node_attr,
        engine='dot')
    g.body.extend(['rankdir=LR'])

    # input nodes
    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')

    # Identify all intermediate nodes involved
    all_intermediate_nodes = set()
    for _, from_node, to_node in edges:
        if from_node not in [-2, -1]:
            all_intermediate_nodes.add(from_node)
        all_intermediate_nodes.add(to_node)

    displayed_nodes_set = set(nodes)
    for node_name in all_intermediate_nodes:
        fill_color = 'lightblue' if node_name in displayed_nodes_set else 'lightgrey'
        g.node(str(node_name), fillcolor=fill_color)

    # Edges 
    for op_name, from_node, to_node in edges:
        if from_node == -2:
            u = "c_{k-2}"
        elif from_node == -1:
            u = "c_{k-1}"
        else:
            u = str(from_node)

        v = str(to_node)
        if op_name != 'none':
            g.edge(u, v, label=op_name, fillcolor="gray")

    # output node
    g.node("c_{k}", fillcolor='palegoldenrod')
    for node_name in nodes:
        g.edge(str(node_name), "c_{k}", fillcolor="gray")

    # add image caption
    if caption:
        g.attr(label=caption, overlap='false', fontsize='20', fontname='times')

    g.render(file_path, view=False)

if __name__ == "__main__":
    Genotype4 = namedtuple('Genotype4', 'normal normal_output_idx reduce reduce_output_idx')
    plotstr="optim"
    genotype = eval("Genotype4(normal=[[('sep_conv_3x3', -2, 0), ('dil_conv_3x3', -1, 0), ('dil_conv_3x3', 0, 1), ('dil_conv_5x5', -2, 2), ('skip_connect', -1, 2), ('sep_conv_3x3', -2, 3), ('skip_connect', 1, 3), ('sep_conv_5x5', 2, 3), ('dil_conv_5x5', -2, 4), ('dil_conv_5x5', 1, 4), ('sep_conv_5x5', 2, 4)], [('skip_connect', -2, 0), ('sep_conv_3x3', -1, 0), ('dil_conv_5x5', -2, 1), ('sep_conv_3x3', -1, 1), ('skip_connect', 0, 2), ('skip_connect', 1, 2), ('dil_conv_3x3', -2, 3), ('sep_conv_5x5', -1, 3), ('sep_conv_3x3', -1, 4), ('max_pool_3x3', 2, 4), ('dil_conv_5x5', 3, 4)], [('skip_connect', -2, 0), ('dil_conv_5x5', -1, 0), ('sep_conv_3x3', -2, 1), ('sep_conv_3x3', -1, 1), ('sep_conv_5x5', -2, 2), ('max_pool_3x3', 0, 2), ('sep_conv_5x5', 1, 3), ('avg_pool_3x3', 2, 3), ('sep_conv_5x5', -2, 4), ('sep_conv_3x3', 2, 4), ('sep_conv_5x5', 3, 4)]], normal_output_idx=[[0, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 4]], reduce=[[('max_pool_3x3', -2, 0), ('sep_conv_5x5', -1, 0), ('skip_connect', -2, 1), ('sep_conv_5x5', -1, 1), ('max_pool_3x3', 0, 2), ('skip_connect', 1, 2), ('max_pool_3x3', -2, 3), ('skip_connect', -1, 3), ('max_pool_3x3', 2, 3), ('sep_conv_5x5', -2, 4), ('sep_conv_3x3', -1, 4)], [('skip_connect', -2, 0), ('skip_connect', -1, 0), ('sep_conv_3x3', 0, 1), ('dil_conv_3x3', -2, 2), ('dil_conv_3x3', -1, 2), ('sep_conv_5x5', 1, 2), ('avg_pool_3x3', -2, 3), ('max_pool_3x3', -1, 3), ('sep_conv_5x5', 1, 3), ('avg_pool_3x3', -2, 4), ('max_pool_3x3', -1, 4)]], reduce_output_idx=[[0, 1, 3, 4], [0, 2, 3, 4]])")
    plot(genotype.normal[0],genotype.normal_output_idx[0], "./SWNAS/" + plotstr + "/n1")
    plot(genotype.normal[1],genotype.normal_output_idx[1], "./SWNAS/" + plotstr + "/n2")
    plot(genotype.normal[2],genotype.normal_output_idx[2], "./SWNAS/" + plotstr + "/n3")
    plot(genotype.reduce[0],genotype.reduce_output_idx[0], "./SWNAS/" + plotstr + "/r1")
    plot(genotype.reduce[1],genotype.reduce_output_idx[1], "./SWNAS/" + plotstr + "/r2")
