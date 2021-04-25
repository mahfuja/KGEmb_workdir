"""Euclidean Knowledge Graph embedding models where embeddings are in real space."""
import numpy as np
import torch
from torch import nn

from models.base import KGModel
from utils.euclidean import euc_sqdistance, givens_rotations, givens_reflection, givens_DE_rotations

EUC_MODELS = ["TransE", "CP", "MurE", "RotE", "RefE", "AttE", "FieldE", "DE"]


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

class FieldE(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""
    def __init__(self, args):  
        super(FieldE, self).__init__(args)                     
        self.hidrank = 500 
        #torch.set_default_dtype(torch.double)
        self.hidden_embedding = nn.Sequential(
                        nn.Linear(self.rank, self.hidrank),                             
                        nn.Tanh(),                                   
                        nn.Linear(self.hidrank,self.hidrank),          
                        nn.Tanh())
                                         
        self.rel_emb = nn.Parameter(2*torch.rand(self.sizes[1], self.hidrank * self.rank) - 1.0) #nn.Embedding(self.sizes[1], self.rank*self.hidrank)
        #self.rel_emb.weight.data = 2 * torch.rand((self.sizes[1], self.rank*self.hidrank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries):
        head_e = self.entity(queries[:, 0])
        rel_e = torch.index_select(                           
                self.rel_emb,
                dim=0,
                index=queries[:, 1]
            )#.unsqueeze(1)


        headHidden = self.hidden_embedding(head_e)            
        relation = rel_e.view(rel_e.size()[0], self.rank, self.hidrank)
        relationh = torch.einsum('ijk,ik->ij', [relation, headHidden])#.squeeze(2)
        relationh1 = torch.tanh(relationh)
        lhs_e = head_e + relationh1
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases


#class DE(BaseE):    
#    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""
    
#    def __init__(self, args):    
#        super(FieldE, self).__init__(args)                
#        assert self.rank % 4 == 0, "DE models require 4D embedding dimension"
#        self.rel_emb = nn.Parameter(2*torch.rand(self.sizes[1], self.rank) - 1.0)
#        self.sim = "dist"
        
#    def get_queries(self, queries):    
#        head_e = self.entity(queries[:, 0])
#        rel_e = torch.index_select(
#                self.rel_emb,                
#                dim=0,                
#                index=queries[:, 1]                
#                )#.unsqueeze(1)
        

#        headHidden = self.hidden_embedding(head_e)        
#        relation = rel_e.view(rel_e.size()[0], self.rank, self.hidrank)        
#        relationh = torch.einsum('ijk,ik->ij', [relation, headHidden])#.squeeze(2)        
#        relationh1 = torch.tanh(relationh)        
#        lhs_e = head_e + relationh1        
#        lhs_biases = self.bh(queries[:, 0])    
#        return lhs_e, lhs_biases



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

class DE(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args):    
        super(DE, self).__init__(args)
        self.rel_rot = nn.Embedding(self.sizes[1], self.rank)
        self.rel_rot.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries):    
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        rel_rot_e = self.rel_rot(queries[:, 1])
        rank = self.rank//2
        
        head_e = head_e[:, :rank//2], head_e[:, rank//2:rank], head_e[:, rank:3*rank//2], head_e[:, 3*rank//2:]
        #rel_e = rel_e[:, :rank//2], rel_e[:, rank//2:rank], rel_e[:, rank:3*rank//2], rel_e[:, 3*rank//2:]
        rel_rot_e = rel_rot_e[:, :rank//2], rel_rot_e[:, rank//2:rank], rel_rot_e[:, rank:3*rank//2], rel_rot_e[:, 3*rank//2:]

        ##s_h = head_e[0]        
        ##x_h = head_e[1] 
        ##y_h = head_e[2]
        ##z_h = head_e[3]

        #s_r = rel_e[0]
        #x_r = rel_e[1]
        #y_r = rel_e[2]
        #z_r = rel_e[3]

        ##s_rot = rel_rot_e[0]        
        ##x_rot = rel_rot_e[1]        
        ##y_rot = rel_rot_e[2]        
        ##z_rot = rel_rot_e[3]

        ##A = s_h * s_rot - x_h * x_rot + y_h * y_rot + z_h * z_rot        
        ##B = s_h * x_rot + s_rot * x_h - y_h * z_rot + y_rot * z_h
        ##C = s_h * y_rot + s_rot * y_h + z_h * x_rot - z_rot * x_h
        ##D = s_h * z_rot + s_rot * z_h + x_h * y_rot - x_rot * y_h

        A, B, C, D = givens_DE_rotations(rel_rot_e, head_e)
        
        E = torch.cat((A, B), 1)
        F = torch.cat((E, C), 1)
        h_e = torch.cat((F, D), 1)
        #print(h_e.size())
        #print()
        #exit()
        lhs_e = h_e + rel_e
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases

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
