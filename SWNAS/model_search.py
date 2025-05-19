import torch
import torch.nn as nn
from operations import *
from genotypes import PRIMITIVES


class OpsAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(OpsAttention, self).__init__()
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        attention_out = x * y.expand_as(x)

        op_attention = []
        op_channel = c // 8 
        for i in range(8):
            temp = y[:, i * op_channel:op_channel * (i + 1), :, :]  
            op_i_atten = torch.sum(temp) 
            op_attention.append(op_i_atten.item())

        output = 0
        for i in range(8):
            output += attention_out[:, i*op_channel:op_channel*(i+1), :, :] 
        return output, op_attention
    
class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.channel = C
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            self._ops.append(op)

        self.attention = OpsAttention(C*8, ratio=8)
  
    def forward(self, x):
        temp = torch.cat([op(x) for op in self._ops], dim=1)  
        return self.attention(temp)
    
class SingleOp(nn.Module):
    def __init__(self, op, C, stride):
        super(SingleOp, self).__init__()
        self.op = OPS[op](C, stride, False)
        op_idx = PRIMITIVES.index(op)
        self.fake_att = [0.0] * len(PRIMITIVES)
        self.fake_att[op_idx] = 1.0

    def forward(self, x):
        return self.op(x), self.fake_att

class NodeAttention(nn.Module):
    def __init__(self, n_nodes, C, reduction=16):
        super().__init__()
        self.n_nodes = n_nodes
        self.in_features = n_nodes * C
        hidden_features = self.in_features // reduction

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, n_nodes),
            nn.Sigmoid()
        )

    def forward(self, x):
        pooled = self.pool(x)
        pooled_flat = pooled.view(pooled.size(0), -1)
        node_attention = self.mlp(pooled_flat)
        return node_attention

class Cell(nn.Module):

    def __init__(self, n_node, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, stage, genotype=None, output_idx = None):
        super(Cell, self).__init__()
        self._n_node = n_node
        self._multiplier = multiplier
        self.C = C
        self.reduction = reduction
        self.reduction_prev = reduction_prev
        self.stage = stage
        self.output_idx = output_idx
        
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        self.dag = nn.ModuleList()
        if genotype is not None:
            for op_name, src_node, dst_node in genotype:
                stride = 2 if reduction and src_node < 0 else 1
                op = SingleOp(op_name, C, stride)
                if len(self.dag) <= dst_node:
                    self.dag.append(nn.ModuleDict())
                self.dag[dst_node][str(src_node)] = op
        else:
            for i in range(self._n_node):
                edges = nn.ModuleDict()
                for src in range(i + 2):
                    stride = 2 if reduction and src < 2 else 1
                    edges[f'{src-2}'] = MixedOp(C, stride)
                self.dag.append(edges)

        self.node_attention = NodeAttention(n_nodes=self._n_node, C=C)

    def forward(self, s0, s1, epoch):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        op_Attention = []
        for dst, edges in enumerate(self.dag):
            node_input_states = []
            op_Attention.append({})
            for src, op in edges.items():
                weighted_op_output, op_attention = op(states[int(src) + 2])
                op_Attention[dst][src] = op_attention
                node_input_states.append(weighted_op_output)
            states.append(sum(node_input_states))
        
        output_nodes = [states[i + 2] for i in self.output_idx]
        concat_output = torch.cat(output_nodes, dim=1)
        if epoch <= 45: # time to use node attention
            b = concat_output.size(0)
            node_attention = torch.ones(b, self._n_node, device=concat_output.device, dtype=concat_output.dtype)
            return concat_output, op_Attention, node_attention

        node_attention = self.node_attention(concat_output)
        
        weighted_nodes = []
        for i, node_output in enumerate(output_nodes):
            weight = node_attention[:, i].view(-1, 1, 1, 1)
            weighted_nodes.append(node_output * weight)

        concatenated_weighted_nodes = torch.cat(weighted_nodes, dim=1)

        return concatenated_weighted_nodes, op_Attention, node_attention


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, n_node=4, multiplier=4, stem_multiplier=3, genotype=None):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        if genotype is not None:
            self.normal_output_idx = genotype.normal_output_idx
            self.reduce_output_idx = genotype.reduce_output_idx
        else:
            self.normal_output_idx = [[i for i in range(n_node)] for _ in range(3)]
            self.reduce_output_idx = [[i for i in range(n_node)] for _ in range(2)]

        C_curr = stem_multiplier * C  # 48
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i < layers // 3:
                stage = 0
                reduction = False
            elif i == layers // 3:
                C_curr *= 2
                stage = 0
                reduction = True
            elif i < (2 * layers) // 3:
                stage = 1
                reduction = False
            elif i == (2 * layers) // 3:
                C_curr *= 2
                stage = 1
                reduction = True
            elif i > (2 * layers) // 3:
                stage = 2
                reduction = False
        
            if genotype is not None:
                genotype_edges = genotype.reduce[stage] if reduction else genotype.normal[stage]
            else:
                genotype_edges = None
            output_idx = self.normal_output_idx[stage] if not reduction else self.reduce_output_idx[stage]
            cell = Cell(n_node, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, stage, genotype_edges, output_idx)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr  # 16 16*4


        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input, epoch):
        s0 = s1 = self.stem(input)
        op_Attention_normal_all = []
        op_Attention_reduce_all = []
        node_weights_normal_all = [] 
        node_weights_reduce_all = [] 

        for i, cell in enumerate(self.cells):
            if cell.reduction:
                s2, op_Attention_reduce, node_weights_reduce = cell(s0, s1, epoch)
                op_Attention_reduce_all.append(op_Attention_reduce)
                node_weights_reduce_all.append(node_weights_reduce) 
            else:
                s2, op_Attention_normal, node_weights_normal = cell(s0, s1, epoch)
                op_Attention_normal_all.append(op_Attention_normal)
                node_weights_normal_all.append(node_weights_normal) 

            s0, s1 = s1, s2
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, op_Attention_normal_all, op_Attention_reduce_all, node_weights_normal_all, node_weights_reduce_all

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)


