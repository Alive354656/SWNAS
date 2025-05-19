import torch
import torch.nn as nn
from operations import *
from genotypes import to_dag


class Cell(nn.Module):
    def __init__(self, genotype, output_idx, C_pp, C_p, C, reduction_p, reduction, stage, drop_p):
        super().__init__()
        self.reduction = reduction
        self.stage = stage

        if reduction_p:
            self.preproc0 = FactorizedReduce(C_pp, C)
        else:
            self.preproc0 = ReLUConvBN(C_pp, C, 1, 1, 0)
        self.preproc1 = ReLUConvBN(C_p, C, 1, 1, 0)

        if self.stage == 0:
            gene = genotype[0]
            self.output_idx = output_idx[0]
        elif self.stage == 1:
            gene = genotype[1]
            self.output_idx = output_idx[1]
        elif self.stage == 2:
            gene = genotype[2]
            self.output_idx = output_idx[2]
        
        self.n_nodes = len(self.output_idx)
        self.dag = to_dag(C, gene, reduction, drop_p)
    
    def forward(self, s0, s1):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)
        states = [s0, s1]

        for edges in self.dag:
            node_input_states = []
            for src_node_idx, op in edges.items():
                weighted_op_output = op(states[int(src_node_idx) + 2])
                node_input_states.append(weighted_op_output)
            states.append(sum(node_input_states))
        
        s_out = [states[i + 2] for i in self.output_idx]
        return torch.cat(s_out, dim=1)



class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x

class NetworkCIFAR(nn.Module):
    def __init__(self, input_size, C_in, C, n_classes, n_layers, auxiliary, genotype, drop_p, stem_multiplier=3):
        super().__init__()
        self.C_in = C_in
        self.C = C                   # 36
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.genotype = genotype
        self.aux_pos = 2 * n_layers // 3 if auxiliary else -1

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            if i < n_layers // 3:
                stage = 0
                reduction = False
            elif i == n_layers // 3:
                C_cur *= 2
                stage = 0
                reduction = True
            elif i < (2 * n_layers) // 3:
                stage = 1
                reduction = False
            elif i == (2 * n_layers) // 3:
                C_cur *= 2
                stage = 1
                reduction = True
            elif i > (2 * n_layers) // 3:
                stage = 2
                reduction = False
        
            edges = genotype.reduce if reduction else genotype.normal
            output_idx = genotype.reduce_output_idx if reduction else genotype.normal_output_idx
            cell = Cell(edges, output_idx, C_pp, C_p, C_cur, reduction_p, reduction, stage, drop_p)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * cell.n_nodes
            C_pp, C_p = C_p, C_cur_out

            if i == self.aux_pos:
                self.aux_head = AuxiliaryHeadCIFAR(input_size // 4, C_p, n_classes)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)
    
    def forward(self, x):
        s0 = s1 = self.stem(x)

        aux_logits = None
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i == self.aux_pos and self.training:
                aux_logits = self.aux_head(s1)
        
        out = self.gap(s1)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits, aux_logits
  
