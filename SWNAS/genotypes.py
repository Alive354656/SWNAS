from collections import namedtuple
import torch.nn as nn
from operations import *

Genotype4 = namedtuple('Genotype4', 'normal normal_output_idx reduce reduce_output_idx')

def parse(genotype):
    return eval(genotype)

PRIMITIVES = [
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'skip_connect',
    'avg_pool_3x3',
    'max_pool_3x3',
    'none'
]

def to_dag(C_in, gene, reduction, drop_p):
    """ generate discrete ops from gene """
    dag = nn.ModuleList()
    for op_name, src_node, dst_node in gene:
        stride = 2 if reduction and src_node < 0 else 1
        op = OPS[op_name](C_in, stride, True)
        if not isinstance(op, Identity) and drop_p > 0: # Identity does not use drop path
            op = nn.Sequential(
                op,
                DropPath(drop_p)
            )
        if len(dag) <= dst_node:
            dag.append(nn.ModuleDict())
        dag[dst_node][str(src_node)] = op

    return dag
