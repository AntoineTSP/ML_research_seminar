import torch
from torch_geometric.loader import DataLoader
from model.GNN_model import GCN
from model.layer_selector import local_pooling_selection, global_pooling_selection, conv_selection
import torch_geometric.nn as nn
import copy
from torch_geometric import datasets
import os
import numpy as np
import json
import yaml
import argparse

def train(model, alpha=1e-2):
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        data=data.to(device)
        out, losses = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y) + alpha*torch.sum(losses)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(model, loader):
    model.eval()
    loss_epoch = []
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data=data.to(device)
        out, losses = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y) + alpha*torch.sum(losses)  # Compute the loss.
        loss_epoch.append(loss.detach().cpu().item())
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset), np.mean(loss_epoch)  # Derive ratio of correct predictions.

def training_loop(nb_max_epochs, patience, verbose=1, alpha=1e-2):
  train_losses = []
  val_losses = []
  train_accuracies = []
  val_accuracies = []
  min_val_acc = -1
  iterations_WO_improvements = 0
  for epoch in range(1, nb_max_epochs):
      train(model, alpha)
      train_acc, train_loss = test(model, train_loader)
      val_acc, val_loss = test(model, val_loader)
      train_losses.append(train_loss)
      val_losses.append(val_loss)
      train_accuracies.append(train_loss)
      val_accuracies.append(val_loss)

      # Early stopping
      if min_val_acc < -1 or min_val_acc < val_acc:
        min_val_acc = val_acc
        iterations_WO_improvements = 0
        best_model = copy.deepcopy(model)
      else:
        iterations_WO_improvements += 1

      if iterations_WO_improvements > patience:
        break

      if verbose>1:
        # Print should be replaced by logs ideally
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

  test_acc, _ = test(best_model, test_loader)
  last_epoch = epoch
  return best_model, test_acc, train_losses, val_losses, train_accuracies, val_accuracies, last_epoch


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process a YAML configuration file and output the results to a JSON file.')
  parser.add_argument('-c', '--config', default="configs\config_test.yml", help='Path to the YAML configuration file')
  args = parser.parse_args()

  file_path=args.config

  # Reading the config file
  with open(file_path, 'r') as file:
      config = yaml.safe_load(file)["model"]

  output_model_path = config.pop("output_model_path", "model/weights")
  output_results_path = config.pop("output_results_path", "model/results")
  device = config.pop("device", None)
  if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  verbose = config.pop("verbose", 2)
  dataset_path = config.pop("dataset_path", "data/TUDataset")
  location = dataset_path.split("/")[-1]
  nb_of_splits = config.pop("nb_of_splits", 10)
  hidden_channels = config.pop("hidden_channels", 64)

  result = dict(config)
  dataset_name = config.pop("dataset")
  max_epochs = config.pop("max_epochs", 200)
  patience = config.pop("patience", 20)
  lr = config.pop("lr", 0.005)
  alpha = float(config.pop("alpha", 1e-2))
  batch_size = config.pop("batch_size", 64)
  conv_layer = config.pop("convolution_layer", "GCN")
  attention_heads = config.pop("attention_heads", 4)
  global_pooling_layer = config.pop("global_pooling_layer", "mean")
  local_pooling_layer = config.pop("local_pooling_layer", "SAG")
    
  use_deterministic_algorithms = config.pop("deterministic_algorithms", True)
  if use_deterministic_algorithms:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

  # Loading the dataset with the string
  dataset = eval("datasets."+location)(dataset_path, name=dataset_name)

  best_test_acc = 0
  test_accuracy_list = [] 

  for i in range(nb_of_splits):
    
    # Creation of the dataset
    torch.manual_seed(12345+i)
    torch.cuda.manual_seed_all(12345+i)
    dataset = dataset.shuffle()
    n = len(dataset)
    train_dataset = dataset[:int(0.6*n)]
    val_dataset = dataset[int(0.6*n):int(0.8*n)]
    test_dataset = dataset[int(0.8*n):]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model build
    local_pooling, dic_conversion_layer = local_pooling_selection(local_pooling_layer, device=device)
    convolutional_layer=conv_selection(conv_layer, attention_heads)

    model = GCN(num_node_features=dataset.num_node_features, 
                num_classes=dataset.num_classes, 
                hidden_channels=hidden_channels,
                conv_method=convolutional_layer, 
                global_pool_method=global_pooling_selection(global_pooling_layer), 
                local_pool_method=local_pooling,
                dic_conversion_layer=dic_conversion_layer).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Model training
    best_model, test_acc, train_losses, val_losses, train_accuracies, val_accuracies, last_epoch = training_loop(max_epochs, patience, alpha=alpha)
    result["split "+str(i+1)] = {"train_losses":train_losses,
                               "val_losses":val_losses,
                               "train_accuracies":train_accuracies,
                               "val_accuracies":val_accuracies,
                               "test_accuracy":test_acc,
                               "last_epoch":last_epoch}
    if verbose > 0:
      print(f'Model number: {i:02d}, Test Acc: {test_acc:.4f}')
    test_accuracy_list.append(test_acc)

  # Compute accuracies and informations about the model
  result["nb_parameters"] = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
  result["mean_accuracy"] = np.mean(test_accuracy_list)
  result["std_accuracy"] = np.std(test_accuracy_list)

  # Model saving
  print(f'Mean Test Acc: {result["mean_accuracy"]:.4f}, Std Test Acc: {result["std_accuracy"]:.4f}')
  
  os.makedirs(output_model_path, exist_ok = True) 

  model_name = f"{dataset_name}_{conv_layer}_{global_pooling_layer}_{local_pooling_layer}"

  torch.save(best_model.state_dict(), os.path.join(output_model_path, model_name))

  os.makedirs(output_results_path, exist_ok = True)

  with open(os.path.join(output_results_path, model_name) + ".json", 'w') as json_file:
      json.dump(result, json_file, indent=2)