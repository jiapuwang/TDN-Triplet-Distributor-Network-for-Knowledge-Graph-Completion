"""Euclidean Knowledge Graph embedding models where embeddings are in real space."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F, Parameter

from models.base import KGModel
from utils.euclidean import euc_sqdistance, givens_rotations, givens_reflection

EUC_MODELS = ["TransE", "CP", "MurE", "RotE", "RefE", "AttE","Incept"]


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

'''
class TransE(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args):
        super(TransE, self).__init__(args)
        self.sim = "dist"

    def get_queries(self, queries):
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        lhs_e = head_e + rel_e
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases
'''

class CP(BaseE):
    """Canonical tensor decomposition https://arxiv.org/pdf/1806.07297.pdf"""

    def __init__(self, args):
        super(CP, self).__init__(args)
        self.sim = "dot"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        return self.entity(queries[:, 0]) * self.rel(queries[:, 1]), self.bh(queries[:, 0])


class MurE(BaseE):
    """Diagonal scaling https://arxiv.org/pdf/1905.09791.pdf"""

    def __init__(self, args):
        super(MurE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        lhs_e = self.rel_diag(queries[:, 1]) * self.entity(queries[:, 0]) + self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases


class RotE(BaseE):
    """Euclidean 2x2 Givens rotations"""

    def __init__(self, args):
        super(RotE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        lhs_e = givens_rotations(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0])) + self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases


class RefE(BaseE):
    """Euclidean 2x2 Givens reflections"""

    def __init__(self, args):
        super(RefE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        lhs = givens_reflection(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0]))
        rel = self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs + rel, lhs_biases


class AttE(BaseE):
    """Euclidean attention model combining translations, reflections and rotations"""

    def __init__(self, args):
        super(AttE, self).__init__(args)
        self.sim = "dist"

        # reflection
        self.ref = nn.Embedding(self.sizes[1], self.rank)
        self.ref.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

        # rotation
        self.rot = nn.Embedding(self.sizes[1], self.rank)
        self.rot.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

        # attention
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.act = nn.Softmax(dim=1)
        self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_reflection_queries(self, queries):
        lhs_ref_e = givens_reflection(
            self.ref(queries[:, 1]), self.entity(queries[:, 0])
        )
        return lhs_ref_e

    def get_rotation_queries(self, queries):
        lhs_rot_e = givens_rotations(
            self.rot(queries[:, 1]), self.entity(queries[:, 0])
        )
        return lhs_rot_e

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        lhs_ref_e = self.get_reflection_queries(queries).view((-1, 1, self.rank))
        lhs_rot_e = self.get_rotation_queries(queries).view((-1, 1, self.rank))

        # self-attention mechanism
        cands = torch.cat([lhs_ref_e, lhs_rot_e], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        lhs_e = torch.sum(att_weights * cands, dim=1) + self.rel(queries[:, 1])
        return lhs_e, self.bh(queries[:, 0])

class Incept(BaseE):

    def __init__(self, args):
        super(Incept, self).__init__(args)
        self.sim = "dist"

        in_planes = 2
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
            
        self.branch4_e1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1,padding=1,  bias=True),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())
        self.branch4_r = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1,padding=1,  bias=True),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())
        self.branch4_s = torch.nn.Sequential(
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
        
        e1 = self.branch2_e1(e1+r+shared)
        r = self.branch2_r(r+e1+shared)
        shared = self.branch2_s(shared+e1+r)
        
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
        
        e1 = self.branch3_e1(e1+r+shared)
        r = self.branch3_r(r+e1+shared)
        shared = self.branch3_s(shared+e1+r)
        
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

        lhs_e = self.fc_e1(e1)+self.fc_r(r)
        #lhs_e = self.fc_e1(shared)
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
