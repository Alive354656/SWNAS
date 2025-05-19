import torch
import torch.optim as optim
import genotypes as gt
import torch.nn as nn
from model_search import MixedOp

class SearchSpaceController:
    def __init__(self, model, args, logger):
        self.model = model
        self.config = args
        self.logger = logger
        self.primitives = gt.PRIMITIVES 

    def insert_node(self, stage, prev_nodes, target_nodes, is_reduce=False):
        cell_type = 1 if is_reduce else 0
        target_cells = [cell for cell in self.model.cells if cell.stage == stage and cell.reduction == is_reduce]
        prev_nodes = sorted(prev_nodes)
        if target_nodes:
            target_nodes = sorted(target_nodes)

        if not target_cells:
            self.logger.warning(f"Not found {cell_type} stage={stage} cell")
            return False

        device = next(self.model.parameters()).device

        if not prev_nodes:
            self.logger.warning("Previous node list cannot be empty")
            return False
        if min(prev_nodes) < -2 or max(prev_nodes) >= target_cells[0]._n_node:
            self.logger.warning("Previous node index out of range")
            return False
        if not target_nodes:
            self.logger.warning("The hidden node to be inserted must have a successor node")
            return False
        if target_nodes:
            if min(target_nodes) < 0 or max(target_nodes) >= target_cells[0]._n_node:
                self.logger.warning("Successor node index out of range")
                return False
            if min(target_nodes) <= max(prev_nodes):
                self.logger.warning("Successor node cannot be less than previous node")
                return False

        insert_idx = max(prev_nodes) + 1
        if is_reduce:
            new_output_idx = [idx if idx < insert_idx else idx + 1 for idx in self.model.reduce_output_idx[stage]]
            self.model.reduce_output_idx[stage] = new_output_idx
        else:
            new_output_idx = [idx if idx < insert_idx else idx + 1 for idx in self.model.normal_output_idx[stage]]
            self.model.normal_output_idx[stage] = new_output_idx
        for cell in target_cells:
            cell._n_node += 1
            cell.output_idx = new_output_idx

            flag = False
            new_dag = nn.ModuleList()
            for dst_node_idx, dst_node_edges in enumerate(cell.dag):    
                if dst_node_idx == insert_idx:
                    flag = True
                    edges = nn.ModuleDict()
                    for src_node_idx in prev_nodes:
                        stride = 2 if cell.reduction and src_node_idx < 0 else 1
                        op = MixedOp(cell.C, stride).to(device)
                        edges[f'{src_node_idx}'] = op
                    new_dag.append(edges)
                if dst_node_idx >= insert_idx:
                    edges = {}
                    for src_node_idx, edge in dst_node_edges.items():
                        if int(src_node_idx) < insert_idx:
                            edges[src_node_idx] = edge
                        else:
                            edges[str(int(src_node_idx)+1)] = edge
                    if dst_node_idx in target_nodes:
                        edges[str(insert_idx)] = MixedOp(cell.C, stride=1).to(device)
                    edges = nn.ModuleDict(sorted(edges.items(), key=lambda x: int(x[0])))
                    new_dag.append(edges)
                else:
                    new_dag.append(dst_node_edges)
            if not flag:
                edges = nn.ModuleDict()
                for src_node_idx in prev_nodes:
                    stride = 2 if cell.reduction and src_node_idx < 0 else 1
                    op = MixedOp(cell.C, stride).to(device)
                    edges[f'{src_node_idx}'] = op
                new_dag.append(edges)
            cell.dag = new_dag

        self.logger.info(f"Successfully inserted hidden node and updated Alpha: {cell_type} stage={stage}")
        return True

    def update_env(self, old_optimizer):
        self.logger.info("Start updating optimizer and learning rate scheduler...")
        w_state_dict = {id(p): s for p, s in old_optimizer.state.items()}

        w_momentum = old_optimizer.param_groups[0]['momentum']
        w_weight_decay = old_optimizer.param_groups[0]['weight_decay']
        w_dampening = old_optimizer.param_groups[0].get('dampening', 0)
        w_nesterov = old_optimizer.param_groups[0].get('nesterov', False)

        new_w_optim = optim.SGD(
            self.model.parameters(),
            lr=0.025,
            momentum=w_momentum,
            weight_decay=w_weight_decay,
            dampening=w_dampening,
            nesterov=w_nesterov
        )

        for group in new_w_optim.param_groups:
            for p in group['params']:
                pid = id(p)
                if pid in w_state_dict:
                    new_w_optim.state[p].update(w_state_dict[pid])

        new_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            new_w_optim,
            T_max=float(self.config.epochs),
            eta_min=self.config.learning_rate_min,
        )

        self.logger.info("Environment update completed")
        return new_w_optim, new_lr_scheduler

        