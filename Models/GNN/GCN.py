import torch
import torch.nn as nn
from Models.GNN.GCN_layer  import GraphConvolution
from Models.GNN.GCN_res_layer import GraphResConvolution
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GCN(nn.Module):
    def __init__(self,
                 state_dim=256,
                 feature_dim=256,
                 out_dim=2,
                 layer_num=8):

        super(GCN, self).__init__()
        self.state_dim = state_dim
        self.layer_num = layer_num

        self.first_gcn = GraphConvolution(feature_dim, 'first_gcn', out_state_dim=self.state_dim)
        self.middle_gcn = nn.ModuleList([])
        for i in range(self.layer_num - 2):
            self.middle_gcn.append(GraphResConvolution(self.state_dim, 'gcn_res_%d' % (i + 1)))
        self.last_gcn = GraphConvolution(self.state_dim , 'last_gcn', out_state_dim=self.state_dim)

        self.fc = nn.Linear(
            in_features=self.state_dim,
            out_features=out_dim,
        )

    def forward(self, x, adj):
        out = F.relu(self.first_gcn(x, adj))
        for m_gcn in self.middle_gcn:
            out = m_gcn(out, adj)

        out = F.relu(self.last_gcn(out, adj))
        out = self.fc(out)
        return out
