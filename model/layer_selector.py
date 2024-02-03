from torch_geometric.nn import GCNConv, GAT, global_mean_pool, global_max_pool, SAGPooling


def global_pooling_selection(layer : str):
    ''' Returns a global pooling layer '''
    if layer == "mean":
        return global_mean_pool
    elif layer == "max":
        return global_max_pool
    else:
        raise NotImplementedError(layer+" global pooling has not been implemented yet. Check if it has been added to the model/layer_selector.py file.")


def local_pooling_selection(layer : str):
    ''' Returns a local pooling layer '''
    if layer == "SAG":
        return SAGPooling
    elif layer is None:
        return None
    else:
        raise NotImplementedError(layer+" local pooling has not been implemented yet. Check if it has been added to the model/layer_selector.py file.")


def conv_selection(layer : str):
    ''' Returns a convolutional layer '''
    if layer == "GCN":
        return GCNConv
    elif layer == "GAT":
        return GAT
    else:
        raise NotImplementedError(layer+" convolutional layer has not been implemented yet. Check if it has been added to the model/layer_selector.py file.")