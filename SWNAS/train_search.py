import os
import sys
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from model_search import Network
from genotypes import PRIMITIVES, parse
from searchspace import SearchSpaceController

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./data/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels') 
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--n_node', type=int, default=4, help='number of nodes')
parser.add_argument('--genotype', type=str, default=None, help='genotype')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
args = parser.parse_args()

args.save = 'results/{}'.format(args.save)
os.makedirs(args.save, exist_ok=True) 

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    args_str = 'args:\n'
    for key, value in vars(args).items():
        args_str += f"{key} = {value}\n"
    logging.info(args_str)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    genotype = parse(args.genotype) if args.genotype else None
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, args.n_node, genotype=genotype)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    
    # check_point = torch.load(args.model_path)
    # model.load_state_dict(check_point['model'])
    # optimizer.load_state_dict(check_point['optim'])
    
    # ssc = SearchSpaceController(model, args, logging)
    # ssc.insert_node(0, [0], [2, 3], False)
    # ssc.insert_node(0, [0, 1], [2], True)
    # ssc.insert_node(1, [0, 1], [3], False)
    # ssc.insert_node(1, [0], [1, 2], True)
    # ssc.insert_node(2, [1, 2], [3], False)

    # optimizer, scheduler = ssc.update_env(optimizer)
    
    train_transform, _ = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)


    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()[0]
        logging.info('epoch [%d] lr %f', epoch+1, lr)

        # training
        train_acc, train_obj = train(train_queue, model, criterion, optimizer, epoch+1)
        logging.info('train epoch [%d] acc %f, loss %f', epoch+1, train_acc, train_obj)

        scheduler.step()

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion, epoch+1)
        logging.info('valid epoch [%d] acc %f, loss %f', epoch+1, valid_acc, valid_obj)

        if epoch > 45:
            torch.save({'model':model.state_dict(),'optim':optimizer.state_dict()}, os.path.join(args.save, f'{epoch+1}.pt'))


def train(train_queue, model, criterion, optimizer, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    model.train()
    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)

        input = input.cuda()
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, _, _, _, _ = model(input, epoch)
        
        loss = criterion(logits, target)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train epoch [%d] step %03d/%03d loss %f top1 %f %% top5 %f %%', epoch, step, len(train_queue), objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    agg_op_norm = None
    agg_op_redu = None
    agg_node_norm = None
    agg_node_redu = None

    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits, op_norm_batch, op_redu_batch, node_norm_batch, node_redu_batch = model(input, epoch)
            loss = criterion(logits, target)

            if agg_op_norm is None:
                agg_op_norm = []
                for cell_op_att_dst in op_norm_batch: # List[Dict[str, List[float]]]
                    agg_cell_data_dst_nodes = []
                    for dst_op_att_dict in cell_op_att_dst: # Dict[str, List[float]]
                        agg_dst_src_dict = {}
                        for src_node_str, op_attns_float in dst_op_att_dict.items():
                            agg_dst_src_dict[src_node_str] = [0.0] * len(op_attns_float)
                        agg_cell_data_dst_nodes.append(agg_dst_src_dict)
                    agg_op_norm.append(agg_cell_data_dst_nodes)
                
                agg_node_norm = [torch.zeros_like(node_tensor[0]) for node_tensor in node_norm_batch] # Node attention init remains
            if agg_op_redu is None:
                agg_op_redu = []
                for cell_op_att_dst in op_redu_batch: # List[Dict[str, List[float]]]
                    agg_cell_data_dst_nodes = []
                    for dst_op_att_dict in cell_op_att_dst: # Dict[str, List[float]]
                        agg_dst_src_dict = {}
                        for src_node_str, op_attns_float in dst_op_att_dict.items():
                            agg_dst_src_dict[src_node_str] = [0.0] * len(op_attns_float)
                        agg_cell_data_dst_nodes.append(agg_dst_src_dict)
                    agg_op_redu.append(agg_cell_data_dst_nodes)

                agg_node_redu = [torch.zeros_like(node_tensor[0]) for node_tensor in node_redu_batch] # Node attention init remains

            # Aggregation for op_Attn_normal
            for cell_idx in range(len(op_norm_batch)):
                cur_cell_op_att_batch = op_norm_batch[cell_idx]
                agg_cell_store = agg_op_norm[cell_idx]
                
                for dst_node_idx in range(len(cur_cell_op_att_batch)):
                    batch_dst_dict = cur_cell_op_att_batch[dst_node_idx]
                    agg_dst_store_dict = agg_cell_store[dst_node_idx]

                    for src_node_str, batch_op_attns_float in batch_dst_dict.items():
                        current_op_agg_list = agg_dst_store_dict[src_node_str]
                        for op_idx in range(len(batch_op_attns_float)):
                            current_op_agg_list[op_idx] += batch_op_attns_float[op_idx]
                    
                agg_node_norm[cell_idx] += torch.sum(node_norm_batch[cell_idx], dim=0)
            
            # Aggregation for op_Attn_reduce
            for cell_idx in range(len(op_redu_batch)):
                cur_cell_op_att_batch = op_redu_batch[cell_idx]
                agg_cell_store = agg_op_redu[cell_idx]

                for dst_node_idx in range(len(cur_cell_op_att_batch)):
                    batch_dst_dict = cur_cell_op_att_batch[dst_node_idx]
                    agg_dst_store_dict = agg_cell_store[dst_node_idx]

                    for src_node_str, batch_op_attns_float in batch_dst_dict.items():
                        current_op_agg_list = agg_dst_store_dict[src_node_str]
                        for op_idx in range(len(batch_op_attns_float)):
                            current_op_agg_list[op_idx] += batch_op_attns_float[op_idx]
                
                agg_node_redu[cell_idx] += torch.sum(node_redu_batch[cell_idx], dim=0)


            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid epoch [%d] step %03d/%03d loss %f top1 %f %% top5 %f %%', epoch, step, len(valid_queue), objs.avg, top1.avg, top5.avg)
    if epoch >=49:
        print_attention(model, agg_op_norm, agg_op_redu, agg_node_norm, agg_node_redu, model.normal_output_idx, model.reduce_output_idx)
    return top1.avg, objs.avg

def print_attention(model, agg_op_norm, agg_op_redu, agg_node_norm, agg_node_redu, norm_out_idx, redu_out_idx):
    with torch.no_grad():
        node_norm = [[node_val.item() / 2.5e4 for node_val in cell_tensor] for cell_tensor in agg_node_norm]
        node_redu = [[node_val.item() / 2.5e4 for node_val in cell_tensor] for cell_tensor in agg_node_redu]

    normal_cell_count = 0
    reduce_cell_count = 0
    atten_info = ""
    for cell_idx in range(len(model.cells)):
        stage = model.cells[cell_idx].stage
        if model.cells[cell_idx].reduction:
            atten_info += attention_str(agg_op_redu[reduce_cell_count], node_redu[reduce_cell_count], 'Reduce', cell_idx, redu_out_idx[stage])
            reduce_cell_count += 1
        else:
            atten_info += attention_str(agg_op_norm[normal_cell_count], node_norm[normal_cell_count], 'Normal', cell_idx, norm_out_idx[stage])
            normal_cell_count += 1
    logging.info(atten_info)

def attention_str(op_att_cell_dst, node_att_cell, cell_type, cell_index, out_node_indices):
    atten_info = f"{cell_type} Cell {cell_index}\n"

    for i in range(len(op_att_cell_dst)):  
        if i in out_node_indices:
            dst_node_w = node_att_cell[out_node_indices.index(i)]
            atten_info += f"  dst_node {i} (output): node_weights:{dst_node_w:.4f}\n"
        else:
            atten_info += f"  dst_node {i}:\n"

        src_op_weights = op_att_cell_dst[i]
        sorted_src_disp_str = sorted(src_op_weights.keys(), key=lambda x: int(x))

        for src_node_disp_str in sorted_src_disp_str:
            op_weights_e = src_op_weights.get(src_node_disp_str)

            weights_indices = [(weight, k) for k, weight in enumerate(op_weights_e)]
            weights_indices.sort(reverse=True)

            max_weight, max_k = weights_indices[0]
            max_op = PRIMITIVES[max_k]
                
            second_weight, second_k = weights_indices[1]
            sec_op = PRIMITIVES[second_k]
            sec_w_str = f"{second_weight:.0f}"
                
            atten_info += f"    src_node {src_node_disp_str}: {max_op} {max_weight:.0f}"
            atten_info += f", {sec_op} {sec_w_str}"
            atten_info += "\n"
            
    return atten_info

if __name__ == '__main__':
    main()

