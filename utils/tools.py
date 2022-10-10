import logging, yaml, argparse
from numpy.core.fromnumeric import trace
import torch
import torch.nn as nn
import os

def set_lr(optimizer, decay_factor):
    for group in optimizer.param_groups:
        group['lr'] = group['lr'] * decay_factor


class Scheduler(object):
    def __init__(self, patience, delta=0, start_from=0, trace_func=print) -> None:
        super().__init__()
        self.patience = patience
        self.delta = delta
        self.start_from = start_from
        self.best_score = None
        self.counter = 0
        self.trace_func = trace_func
        self.early_stop = False

    def __call__(self, score, epoch):
        if epoch < self.start_from:
            self.early_stop = False
        
        if self.best_score is None:
            self.best_score = score
            self.early_stop = False
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def get_log(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting)
    return config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/base.yaml', type=str, help='path to the config file')
    parser.add_argument('--rank', type=int, default=0, help='rank')
    parser.add_argument('--world_size', type=int, default=1, help='world size')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--dist_url', default='tcp://localhost:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument("--vh_weight", type=float, default=0.5, help="")
    parser.add_argument("--seed", type=float, default=1234, help="")
    parser.add_argument("--use_gpu", type=bool, default=True, help="")
    parser.add_argument("--gpu", type=int, default=0, help="")
    parser.add_argument("--clean", action="store_true", help="")

    cfg = parser.parse_args()
    args = get_config(cfg.config)
    update_values(args, vars(cfg))
    cfg.discriminator_learning_rate = float(cfg.discriminator_learning_rate)
    cfg.generator_learning_rate = float(cfg.generator_learning_rate)
    cfg.pretrain_discriminator_learning_rate = float(cfg.pretrain_discriminator_learning_rate)
    cfg.pretrain_generator_learning_rate = float(cfg.pretrain_generator_learning_rate)

    checkpoint_path = os.path.join(cfg.checkpoint_path, cfg.name)
    cfg.save_dir = os.path.join(cfg.save_dir, cfg.name)
    cfg.generator_pretrain_checkpoint_path = os.path.join(checkpoint_path, "pretrain_generator")
    cfg.discriminator_pretrain_checkpoint_path = os.path.join(checkpoint_path, "pretrain_discriminator")
    cfg.generator_checkpoint_path = os.path.join(checkpoint_path, "generator")
    cfg.discriminator_checkpoint_path = os.path.join(cfg.checkpoint_path, "discriminator")
    

    return cfg


def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]


def adj_dense2edge_sparse(adj: torch.Tensor):
    """

    Parameters
    ----------
    adj: torch.Tensor, shape=[Batch, Node_Number, Node_Number]

    Returns
    -------

    """
    assert len(adj.shape) == 3
    assert adj.shape[1] == adj.shape[2]
    device = adj.device
    max_number = adj.shape[1] * adj.shape[2]
    node_number = adj.shape[1]
    node2edge = torch.zeros(adj.shape[0], max_number, node_number)
    edge2node = torch.zeros(adj.shape[0], node_number, max_number)
    batch_max_edge_number = -1
    for i in range(adj.shape[0]):
        src, tgt = adj[i, :, :].nonzero(as_tuple=True)
        num_edges = src.shape[0]
        batch_max_edge_number = max(batch_max_edge_number, num_edges)
        edge_index = torch.Tensor(range(0, num_edges))
        src_indice = torch.zeros(num_edges, node_number)
        src_indice[edge_index.long(), src] = 1

        tgt_indice = torch.zeros(num_edges, node_number)
        tgt_indice[edge_index.long(), tgt] = 1

        node2edge[i, :num_edges, :] = tgt_indice
        edge2node[i, :, :num_edges] = src_indice.transpose(0, 1)
    node2edge = node2edge[:, :batch_max_edge_number, :]
    edge2node = edge2node[:, :, :batch_max_edge_number]
    node2edge = node2edge.to(device)
    edge2node = edge2node.to(device)

    return node2edge, edge2node


if __name__ == "__main__":
    a = torch.zeros(1, 3, 3)
    a[0, 0, 1] = 1
    a[0, 1, 2] = 1
    adj_dense2edge_sparse(a)
