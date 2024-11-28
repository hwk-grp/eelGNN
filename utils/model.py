import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import FastRGCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool


def model_builder(models, gnn_name, n_atom_feats, num_rel, dim, n_layers, ensemble, aggr, pool):

    if gnn_name == 'eelgcn':
        models.append(eelGCN(n_atom_feats, num_rel, dim, n_layers, aggr, pool))
    elif gnn_name == 'gcn_multi':
        models.append(GCN_multi(n_atom_feats, dim, n_layers, ensemble, aggr, pool))
    elif gnn_name == 'gcn_single':
        models.append(GCN_single(n_atom_feats, dim, n_layers[0], aggr, pool))
    else:
        print('Model {} is not available!'.format(gnn_name))
        exit()

    return models

class eelGCN(nn.Module):
    # Relational GCN (RGCN), Modeling Relational Data with Graph Convolutional Networks. arxiv.org/pdf/1703.06103.pdf
    def __init__(self, num_node_feats, num_rel, dim, n_layers, aggregation, pooling):
        super(eelGCN, self).__init__()
        self.n_layers = n_layers
        self.pool = pooling

        self.cgc = nn.ModuleList()
        self.cgc.append(GCNConv(num_node_feats, dim, normalize=False, bias=False, aggr=aggregation))
        [self.cgc.append(GCNConv(dim, dim, normalize=False, bias=False, aggr=aggregation)) for _ in range(n_layers[0] - 2) if n_layers[0] > 2]
        self.cgc.append(GCNConv(dim, int(dim / 2), normalize=False, bias=False, aggr=aggregation))

        self.sgc = nn.ModuleList()
        self.sgc.append(GCNConv(num_node_feats, dim, normalize=False, bias=False, aggr=aggregation))
        [self.sgc.append(GCNConv(dim, dim, normalize=False, bias=False, aggr=aggregation)) for _ in range(n_layers[1] - 2) if n_layers[1] > 2]
        self.sgc.append(GCNConv(dim, int(dim / 2), normalize=False, bias=False,  aggr=aggregation))

        self.igc = nn.ModuleList()
        self.igc.append(
            FastRGCNConv(num_node_feats, dim, num_rel, root_weight=False, is_sorted=True, bias=False, aggr=aggregation))
        [self.igc.append(
            FastRGCNConv(dim, dim, num_rel, root_weight=False, is_sorted=True, bias=False, aggr=aggregation)) for _
         in range(0, n_layers[2] - 2) if n_layers[2] > 2]
        self.igc.append(FastRGCNConv(dim, int(dim / 2), num_rel, root_weight=False, is_sorted=True, bias=False, aggr=aggregation))

        self.cfc = nn.Linear(int(dim / 2), int(dim / 2))
        self.sfc = nn.Linear(int(dim / 2), int(dim / 2))
        self.icfc = nn.Linear(int(dim / 2), int(dim / 2))
        self.isfc = nn.Linear(int(dim / 2), int(dim / 2))

        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, 1)


    def forward(self, g, pseudo_batch):
        for i in range(self.n_layers[0]):
            if i == 0: c = g.chr_x
            c = F.relu(self.cgc[i](c, g.chr_edge_index))

        for i in range(self.n_layers[1]):
            if i == 0: s = g.slv_x
            s = F.relu(self.sgc[i](s, g.slv_edge_index))

        for i in range(self.n_layers[2]):
            if i == 0: h = g.x
            h = F.relu(self.igc[i](h, g.edge_index, g.edge_type))

        # pseudo_batch is used to pooling chromophore and solvent separately from merged system
        if self.pool == 'add':
            cg, sg, hg = global_add_pool(c, g.chr_x_batch), global_add_pool(s, g.slv_x_batch), global_add_pool(h, pseudo_batch)
        else:
            cg, sg, hg = global_mean_pool(c, g.chr_x_batch), global_mean_pool(s, g.slv_x_batch), global_mean_pool(h, pseudo_batch)

        chg = hg[0::2]
        shg = hg[1::2]

        cg = F.relu(self.cfc(cg))
        sg = F.relu(self.sfc(sg))
        chg = F.relu(self.icfc(chg))
        shg = F.relu(self.isfc(shg))

        hg = torch.cat((cg, sg, chg, shg), dim=1)
        hg = F.relu(self.fc1(hg))
        out = self.fc2(hg)

        return out

class GCN_multi(nn.Module):
    # Relational GCN (RGCN), Modeling Relational Data with Graph Convolutional Networks. arxiv.org/pdf/1703.06103.pdf
    def __init__(self, num_node_feats, dim, n_layers, ensemble, aggregation, pooling):
        super(GCN_multi, self).__init__()
        self.n_layers = n_layers
        self.ensemble = ensemble
        self.pool = pooling

        self.hypercgc = nn.ModuleList()
        for _ in range(ensemble[0]):
            self.cgc = nn.ModuleList()
            self.cgc.append(GCNConv(num_node_feats, dim, normalize=False, bias=False, aggr=aggregation))
            [self.cgc.append(GCNConv(dim, dim, normalize=False, bias=False, aggr=aggregation)) for _ in range(n_layers[0] - 2) if n_layers[0] > 2]
            self.cgc.append(GCNConv(dim, int(dim / 2), normalize=False, bias=False, aggr=aggregation))
            self.hypercgc.append(self.cgc)


        self.hypersgc = nn.ModuleList()
        for _ in range(ensemble[1]):
            self.sgc = nn.ModuleList()
            self.sgc.append(GCNConv(num_node_feats, dim, normalize=False, bias=False, aggr=aggregation))
            [self.sgc.append(GCNConv(dim, dim, normalize=False, bias=False, aggr=aggregation)) for _ in range(n_layers[1] - 2) if n_layers[1] > 2]
            self.sgc.append(GCNConv(dim, int(dim / 2), normalize=False, bias=False, aggr=aggregation))
            self.hypersgc.append(self.sgc)


        self.cfc = nn.ModuleList()
        [self.cfc.append(nn.Linear(int(dim / 2), dim)) for _ in range(ensemble[0])]
        self.sfc = nn.ModuleList()
        [self.sfc.append(nn.Linear(int(dim / 2), dim)) for _ in range(ensemble[1])]

        self.fc1 = nn.Linear(int((ensemble[0] + ensemble[1]) * dim), dim)
        self.fc2 = nn.Linear(dim, 1)


    def forward(self, g, pseudo_batch):

        rep_chr = []
        rep_slv = []

        for n_chr in range(self.ensemble[0]):
            for i in range(self.n_layers[0]):
                if i == 0: c = g.chr_x
                c = F.relu(self.hypercgc[n_chr][i](c, g.chr_edge_index))
            cg = global_add_pool(c, g.chr_x_batch) if self.pool == 'add' else global_mean_pool(c, g.chr_x_batch)
            rep_chr.append(cg)

        for n_slv in range(self.ensemble[1]):
            for i in range(self.n_layers[1]):
                if i == 0: s = g.slv_x
                s = F.relu(self.hypersgc[n_slv][i](s, g.slv_edge_index))
            sg = global_add_pool(s, g.slv_x_batch) if self.pool == 'add' else global_mean_pool(s, g.slv_x_batch)
            rep_slv.append(sg)

        ind_chr = []
        ind_slv = []

        [ind_chr.append(F.relu(self.cfc[i](rep_chr[i]))) for i in range(self.ensemble[0])]
        [ind_slv.append(F.relu(self.sfc[i](rep_slv[i]))) for i in range(self.ensemble[1])]

        hg = torch.cat(ind_chr + ind_slv, dim=1)
        hg = F.relu(self.fc1(hg))
        out = self.fc2(hg)

        return out


class GCN_single(nn.Module):
    def __init__(self, num_node_feats, dim, n_layers, aggregation, pooling):
        super(GCN_single, self).__init__()
        self.n_layers = n_layers
        self.pooling = pooling

        self.gc = nn.ModuleList()
        self.gc.append(GCNConv(num_node_feats, dim, normalize=False, bias=False, aggr=aggregation))
        [self.gc.append(GCNConv(dim, dim, normalize=False, bias=False, aggr=aggregation)) for _ in range(n_layers - 2)
         if n_layers > 2]
        self.gc.append(GCNConv(dim, dim, normalize=False, bias=False, aggr=aggregation))

        self.fc = nn.ModuleList()
        [self.fc.append(nn.Linear(dim, dim)) for _ in range(n_layers - 1) if n_layers > 1]
        self.fc.append(nn.Linear(dim, 1))

    def forward(self, g):

        for i in range(self.n_layers):
            if i == 0: c = g.x
            c = F.relu(self.gc[i](c, g.edge_index))

        cg = global_add_pool(c, g.batch) if self.pooling == 'add' else global_mean_pool(c, g.batch)

        for i in range(self.n_layers):
            cg = F.relu(self.fc[i](cg)) if i != int(self.n_layers) - 1 else self.fc[2](cg)

        return cg
