import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool

class GCN(nn.Module):
    ''' Generic Graph Convolution Network which convolutional layers and pooling layers can be changed'''
    def __init__(self, 
                 num_node_features, 
                 num_classes, 
                 hidden_channels=128, 
                 conv_method = GCNConv, 
                 global_pool_method=global_mean_pool, 
                 local_pool_method=None,
                 dic_conversion_layer=None):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = conv_method(num_node_features, hidden_channels)
        self.conv1_bis = conv_method(hidden_channels, hidden_channels)
        self.conv2 = conv_method(hidden_channels, hidden_channels)
        self.conv2_bis = conv_method(hidden_channels, hidden_channels)
        self.conv3 = conv_method(hidden_channels, hidden_channels)

        self.bn1 = torch_geometric.nn.BatchNorm(hidden_channels)
        self.bn1_bis = torch_geometric.nn.BatchNorm(hidden_channels)
        self.bn2 = torch_geometric.nn.BatchNorm(hidden_channels)
        self.bn2_bis = torch_geometric.nn.BatchNorm(hidden_channels)
        self.bn3 = torch_geometric.nn.BatchNorm(hidden_channels)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)

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
        x = self.bn1(x)
        #x = x.relu()
        x = F.leaky_relu(x, 0.1)

        x = self.dropout1(x)
        
        # x = self.conv1_bis(x, edge_index)
        # x = self.bn1_bis(x)
        # #x = x.relu()
        # x = F.leaky_relu(x, 0.1)
        
        if not self.local_pool_method is None:
            output = self.pool1(x=x, edge_index=edge_index, batch=batch)
            x = output[self.dic_conversion_layer['node_features']]
            edge_index = output[self.dic_conversion_layer['edge_index']]
            batch = output[self.dic_conversion_layer['batch']]
            if not self.dic_conversion_layer.get('loss') is None:
                losses.append(output[self.dic_conversion_layer['loss']])

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        #x = x.relu()
        x = F.leaky_relu(x, 0.1)

        x = self.dropout2(x)
        
        # x = self.conv2_bis(x, edge_index)
        # x = self.bn2_bis(x)
        # #x = x.relu()
        # x = F.leaky_relu(x, 0.1)
        
        if not self.local_pool_method is None:
            output = self.pool2(x=x, edge_index=edge_index, batch=batch)
            x = output[self.dic_conversion_layer['node_features']]
            edge_index = output[self.dic_conversion_layer['edge_index']]
            batch = output[self.dic_conversion_layer['batch']]
            if not self.dic_conversion_layer.get('loss') is None:
                losses.append(output[self.dic_conversion_layer['loss']])

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        #x = x.relu()
        x = F.leaky_relu(x, 0.1)
        x = self.dropout3(x)
        
        if not self.local_pool_method is None:
            output = self.pool3(x=x, edge_index=edge_index, batch=batch)
            x = output[self.dic_conversion_layer['node_features']]
            edge_index = output[self.dic_conversion_layer['edge_index']]
            batch = output[self.dic_conversion_layer['batch']]
            if not self.dic_conversion_layer.get('loss') is None:
                losses.append(output[self.dic_conversion_layer['loss']])

        # 2. Readout layer
        x = self.pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x)

        return x, torch.Tensor(losses)