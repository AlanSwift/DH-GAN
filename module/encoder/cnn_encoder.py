import torch
import torch.nn as nn
import torch.nn.functional as F
from module.submodule.resnet import resnet


class CNNEncoder(nn.Module):
    def __init__(self, cnn_backend, cnn_weight_path, fixed_block, dim_cnn, dim_out=None, dropout=0.2):
        super(CNNEncoder, self).__init__()
        if cnn_backend == "resnet101":
            self.cnn = resnet(cnn_weight_path, _num_layers=101, _fixed_block=fixed_block, pretrained=True)
        else:
            raise NotImplementedError()
        self.dim_cnn = dim_cnn
        self.dim_out = dim_out
        self.require_projection = dim_out is not None
        if self.require_projection:
            self.prj = nn.Sequential(nn.Linear(self.dim_cnn, self.dim_out),
                                     nn.ReLU(),
                                     nn.Dropout(dropout))

    def forward(self, image):
        conv_feats = self.cnn(image)
        batch_size, dim_cnn, step1, step2 = conv_feats.shape
        assert dim_cnn == self.dim_cnn
        conv_feats = conv_feats.view(batch_size, dim_cnn, -1).transpose(1, 2).contiguous()
        if self.require_projection:
            conv_feats = self.prj(conv_feats)
        fc_feats = conv_feats.mean(1)
        return conv_feats, fc_feats
