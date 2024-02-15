from torch_geometric.nn import GCNConv, GAT, global_mean_pool, global_max_pool, SAGPooling
from model.mewisPool_layer import MEWISPool

def global_pooling_selection(layer : str):
    ''' Returns a global pooling layer '''
    if layer == "mean":
        return global_mean_pool
    elif layer == "max":
        return global_max_pool
    else:
        raise NotImplementedError(layer+" global pooling has not been implemented yet. Check if it has been added to the model/layer_selector.py file.")


def local_pooling_selection(layer : str, device : str):
    ''' Returns a local pooling layer '''
    if layer == "SAG":
        dic_conversion_layer = {'node_features':0,'edge_index':1,'batch':3}
        return lambda h:SAGPooling(h), dic_conversion_layer
    elif layer == "MEWIS":
        dic_conversion_layer = {'node_features':0,'edge_index':1,'batch':2, 'loss':3}
        return lambda h:MEWISPool(h, device=device), dic_conversion_layer
    elif layer is None:
        return None, None
    else:
        raise NotImplementedError(layer+" local pooling has not been implemented yet. Check if it has been added to the model/layer_selector.py file.")


def conv_selection(layer : str, attention_heads:int):
    ''' Returns a convolutional layer '''
    if layer == "GCN":
        return GCNConv
    elif layer == "GAT":
        return lambda in_channels,hidden_channels:GAT(in_channels,hidden_channels,num_layers=attention_heads)
    else:
        raise NotImplementedError(layer+" convolutional layer has not been implemented yet. Check if it has been added to the model/layer_selector.py file.")