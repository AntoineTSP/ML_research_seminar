import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(nn.Module):
    ''' Generic Graph Convolution Network which convolutional layers and pooling layers can be changed'''
    def __init__(self, 
                 num_node_features, 
                 num_classes, 
                 hidden_channels=64, 
                 conv_method = GCNConv, 
                 global_pool_method=global_mean_pool, 
                 local_pool_method=None):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = conv_method(num_node_features, hidden_channels)
        self.conv2 = conv_method(hidden_channels, hidden_channels)
        self.conv3 = conv_method(hidden_channels, hidden_channels)

        self.local_pool_method = local_pool_method
        if not local_pool_method is None:
            self.pool1 = local_pool_method(hidden_channels)
            self.pool2 = local_pool_method(hidden_channels)
            self.pool3 = local_pool_method(hidden_channels)

        self.lin = nn.Linear(hidden_channels, num_classes)
        self.pool = global_pool_method

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        if not self.local_pool_method is None:
            x, edge_index, _, batch, perm, score  = self.pool1(x,edge_index, None, batch)

        x = self.conv2(x, edge_index)
        x = x.relu()
        if not self.local_pool_method is None:
            x, edge_index, _, batch, perm, score  = self.pool2(x,edge_index, None, batch)

        x = self.conv3(x, edge_index)
        if not self.local_pool_method is None:
            x, edge_index, _, batch, perm, score  = self.pool3(x,edge_index, None, batch)
            
        # 2. Readout layer
        x = self.pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x