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
                 local_pool_method=None,
                 dic_conversion_layer=None):
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

        self.dic_conversion_layer = dic_conversion_layer

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        losses = []
        x = self.conv1(x, edge_index)
        x = x.relu()
        if not self.local_pool_method is None:
            output = self.pool1(x=x, edge_index=edge_index, batch=batch)
            x = output[self.dic_conversion_layer['node_features']]
            edge_index = output[self.dic_conversion_layer['edge_index']]
            batch = output[self.dic_conversion_layer['batch']]
            if not self.dic_conversion_layer.get('loss') is None:
                losses.append(output[self.dic_conversion_layer['loss']])

        x = self.conv2(x, edge_index)
        x = x.relu()
        if not self.local_pool_method is None:
            output = self.pool1(x=x, edge_index=edge_index, batch=batch)
            x = output[self.dic_conversion_layer['node_features']]
            edge_index = output[self.dic_conversion_layer['edge_index']]
            batch = output[self.dic_conversion_layer['batch']]
            if not self.dic_conversion_layer.get('loss') is None:
                losses.append(output[self.dic_conversion_layer['loss']])

        x = self.conv3(x, edge_index)
        if not self.local_pool_method is None:
            output = self.pool1(x=x, edge_index=edge_index, batch=batch)
            x = output[self.dic_conversion_layer['node_features']]
            edge_index = output[self.dic_conversion_layer['edge_index']]
            batch = output[self.dic_conversion_layer['batch']]
            if not self.dic_conversion_layer.get('loss') is None:
                losses.append(output[self.dic_conversion_layer['loss']])

        # 2. Readout layer
        x = self.pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x, torch.Tensor(losses)