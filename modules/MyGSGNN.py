import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.nn import MessagePassing

__all__ = ['MyGSGNN']


class _MP(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x: Tensor, edge_index: Tensor):
        return self.propagate(edge_index, x=x)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j


class LocalLayer(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.MP_pos = _MP()
        self.MP_neg = _MP()
        self.lin = nn.Linear(in_channels * 3, out_channels)

    def reset_parameters(self):
        self.MP_pos.reset_parameters()
        self.MP_neg.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, pos_edge_index, neg_edge_index):
        x_pos = self.MP_pos(x, pos_edge_index)
        x_neg = self.MP_neg(x, neg_edge_index)
        out_cat = torch.cat((x, x_pos, x_neg), dim=1)
        return self.lin(out_cat)


class GlobalLayer(nn.Module):
    def __init__(self, k) -> None:
        super().__init__()
        self.MP_pos = _MP()
        self.MP_neg = _MP()
        self.k = k
        self.trans = nn.Sequential()
        self.trans.add_module('lin1',nn.Linear(k*3, 16))
        self.trans.add_module('tanh',nn.Tanh())
        self.trans.add_module('lin2',nn.Linear(16,k))

    def reset_parameters(self):
        self.MP_pos.reset_parameters()
        self.MP_neg.reset_parameters()
        for name, module in self.trans.named_modules():
            if 'lin' in name:
                module.reset_parameters()

    def forward(self, x, pos_edge_index, neg_edge_index):
        x_pos = self.MP_pos(x, pos_edge_index)
        x_neg = self.MP_neg(x, neg_edge_index)
        out_cat = torch.cat((x, x_pos, x_neg), dim=1)

        C = self.trans(out_cat)
        return F.softmax(C, dim=1)

class MyGSGNN(nn.Module):
    def __init__(self, in_dims, hid_dims, color_dims=16, G_layers=5, L_layers=2, k=3) -> None:
        super().__init__()

        self.G_layers = G_layers
        self.Globals = nn.ModuleList()
        for i in range(G_layers):
            self.Globals.append(GlobalLayer(k))
        
        self.color = nn.Sequential()
        self.color.add_module('lin0',nn.Linear(in_dims, in_dims//2))
        self.color.add_module('relu',nn.ReLU())
        self.color.add_module('lin1',nn.Linear(in_dims // 2, color_dims))

        self.Cx = nn.Parameter(torch.FloatTensor(k, color_dims))

        self.L_layers = L_layers
        self.Locals = nn.ModuleList()
        self.Locals.append(LocalLayer(in_dims,hid_dims-color_dims))
        for i in range(L_layers - 1):
            self.Locals.append(LocalLayer(hid_dims-color_dims,hid_dims-color_dims))

        self.act = nn.Tanh()

        self.Lins = nn.Sequential()
        self.Lins.add_module('lin0',nn.Linear(hid_dims*2, hid_dims, bias=False))
        self.Lins.add_module('relu0',nn.ReLU())
        self.Lins.add_module('lin1',nn.Linear(hid_dims, hid_dims//2, bias=False))
        self.Lins.add_module('relu1',nn.ReLU())
        self.Lins.add_module('lin2',nn.Linear(hid_dims//2, 2, bias=False))

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.G_layers):
            self.Globals[i].reset_parameters()

        for name, module in self.color.named_modules():
            if 'lin' in name:
                module.reset_parameters()

        nn.init.xavier_normal_(self.Cx)
        
        for i in range(self.L_layers):
            self.Locals[i].reset_parameters()

        for name, module in self.Lins.named_modules():
            if 'lin' in name:
                module.reset_parameters()

    def forward(self, x, pos_edge_index, neg_edge_index):
        cc = self.color(x) # 映射到颜色域上

        CX = cc @ self.Cx.T # 分类
        ck = F.softmax(CX, dim=1)

        G = ck
        for i in range(self.G_layers):
            G = self.Globals[i](G, pos_edge_index, neg_edge_index)
        G = G @ self.Cx

        L = x
        for i in range(self.L_layers):
            L = self.Locals[i](L, pos_edge_index, neg_edge_index)
            if i != self.L_layers - 1:
                L = self.act(L)
        
        return torch.cat([G,L], dim=1)  
    
    def discriminate(self, z, edge_index):
        return self.Lins(torch.cat((z[edge_index[0]], z[edge_index[1]]), dim=1))

    def loss(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat((pos_edge_index,neg_edge_index), dim=1)
        pred_y = self.discriminate(z, edge_index)
        y = torch.cat((torch.zeros(pos_edge_index.size(1)), torch.ones(neg_edge_index.size(1)))).long().to(z.device)
        return F.cross_entropy(pred_y, y)
