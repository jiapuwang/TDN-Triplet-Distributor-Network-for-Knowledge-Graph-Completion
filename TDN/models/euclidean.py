"""Euclidean Knowledge Graph embedding models where embeddings are in real space."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F, Parameter

from models.base import KGModel
from utils.euclidean import euc_sqdistance, givens_rotations, givens_reflection

EUC_MODELS = ["InceptE"]


class BaseE(KGModel):
    """Euclidean Knowledge Graph Embedding models.

    Attributes:
        sim: similarity metric to use (dist for distance and dot for dot product)
    """

    def __init__(self, args):
        super(BaseE, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        if self.sim == "dot":
            if eval_mode:
                score = lhs_e @ rhs_e.transpose(0, 1)
            else:
                score = torch.sum(lhs_e * rhs_e, dim=-1, keepdim=True)
        else:
            score = - euc_sqdistance(lhs_e, rhs_e, eval_mode)
        return score

class InceptE(BaseE):

    def __init__(self, args):
        super(InceptE, self).__init__(args)
        self.sim = "dot"

        in_planes = 64
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()
        
        self.branch1_e1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=1, stride=1, bias=True),
            torch.nn.BatchNorm2d(32))
        self.branch1_r = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=1, stride=1, bias=True),
            torch.nn.BatchNorm2d(32))
            
        self.branch2_e1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())
        self.branch2_r = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())
        self.branch2_s = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())
            
        self.branch3_e1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1,padding=1,  bias=True),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())
        self.branch3_r = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1,padding=1,  bias=True),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())
        self.branch3_s = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1,padding=1,  bias=True),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())

        self.e1_fuse_1x1conv = torch.nn.Conv2d(32, 32, kernel_size=1)
        self.r_fuse_1x1conv = torch.nn.Conv2d(32, 32, kernel_size=1)
        self.e1_distribute_1x1conv = torch.nn.Conv2d(32, 32, kernel_size=1)
        self.r_distribute_1x1conv = torch.nn.Conv2d(32, 32, kernel_size=1)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm1d(200)
        self.bn2 = torch.nn.BatchNorm1d(200)
        self.bn3 = torch.nn.BatchNorm1d(200)
        self.fc_e1 = torch.nn.Linear(6400, 200)
        self.fc_r = torch.nn.Linear(6400, 200)


    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        #c = F.softplus(self.c[queries[:, 1]])
        e1 = self.entity(queries[:, 0])
        r = self.rel(queries[:, 1])
        
        e1 = e1.view(-1, 1, 10, 20)
        r = r.view(-1, 1, 10, 20)
        
        e = torch.cat((e1,r),1)
        out = self.ca(e)*e
        out = self.sa(out)*out
        out = F.relu(out)
        x = torch.add(out, e)
        e1,r = x.chunk(2,1)
        
        e1 = self.branch1_e1(e1)
        r = self.branch1_r(r)
        
        shared = torch.zeros(e1.shape).cuda()
        e1_s = self.e1_fuse_1x1conv(e1 - shared)
        e1_fuse_gate = torch.sigmoid(e1_s)
        r_s = self.r_fuse_1x1conv(r- shared)
        r_fuse_gate = torch.sigmoid(r_s)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate

        s_e1 = self.e1_distribute_1x1conv(shared - e1)
        e1_distribute_gate = torch.sigmoid(s_e1)
        s_r = self.r_distribute_1x1conv(shared - r)
        r_distribute_gate = torch.sigmoid(s_r)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        
        e1 = self.branch2_e1(e1)
        r = self.branch2_r(r)
        shared = self.branch2_s(shared)
        
        e1_s = self.e1_fuse_1x1conv(e1 - shared)
        e1_fuse_gate = torch.sigmoid(e1_s)
        r_s = self.r_fuse_1x1conv(r- shared)
        r_fuse_gate = torch.sigmoid(r_s)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate

        s_e1 = self.e1_distribute_1x1conv(shared - e1)
        e1_distribute_gate = torch.sigmoid(s_e1)
        s_r = self.r_distribute_1x1conv(shared - r)
        r_distribute_gate = torch.sigmoid(s_r)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        
        e1 = self.branch3_e1(e1)
        r = self.branch3_r(r)
        shared = self.branch3_s(shared)
        
        e1_s = self.e1_fuse_1x1conv(e1 - shared)
        e1_fuse_gate = torch.sigmoid(e1_s)
        r_s = self.r_fuse_1x1conv(r- shared)
        r_fuse_gate = torch.sigmoid(r_s)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate
        
        s_e1 = self.e1_distribute_1x1conv(shared - e1)
        e1_distribute_gate = torch.sigmoid(s_e1)
        s_r = self.r_distribute_1x1conv(shared - r)
        r_distribute_gate = torch.sigmoid(s_r)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate

        e1 = e1.view(e1.size(0),-1)
        r = r.view(r.size(0),-1)
        #shared = shared.view(shared.size(0),-1)

        lhs_e = self.fc_e1(e1) * self.fc_r(r)
        return lhs_e, self.bh(queries[:, 0])
        
class ChannelAttention(torch.nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
           
        self.fc = torch.nn.Sequential(torch.nn.Conv2d(in_planes, in_planes // 2, 1, bias=False),
                               torch.nn.ReLU(),
                               torch.nn.Conv2d(in_planes // 2, in_planes, 1, bias=False))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out 
        return self.sigmoid(out)

class SpatialAttention(torch.nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = torch.nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
