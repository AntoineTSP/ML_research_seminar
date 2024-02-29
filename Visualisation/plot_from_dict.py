from typing import List, Dict, Tuple

import warnings

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.markers as markers
import matplotlib.patches as mpatches
from matplotlib import MatplotlibDeprecationWarning

import pandas as pd

def plot_from_dict(list_dict : List[Dict], figsize : Tuple[int,int]) -> None :
  """
  Plot the graph resulting from the list of dictionnary

  Raises an error if the desired keys are not present in a dictionnary
  """
  # checking that there is no problem of keys in each given dictionnary
  key_values_to_check = ['nb_parameters',
                         'mean_accuracy',
                         'homophily',
                         'global_pooling_layer',
                         'convolution_layer',
                         'dataset']

  for i,d in enumerate(list_dict) :

    if not(all(key in d for key in key_values_to_check)) :

      raise Exception(f"Problem of key for the {i}-th dictionnary")


      # the x,y and z of the scatter in 3D
      x = np.array([d['nb_parameters'] for d in list_dict])
      y = np.array([d['mean_accuracy'] for d in list_dict])
      z = np.array([d['homophily'] for d in list_dict])


    # the poolings will be used for the color of the points
    poolings = [d['pooling'] for d in list_dict]
    unique_poolings = np.unique(poolings)

    # all the colors in matplotlib
    # the functions used is deprecated but works in a simpler way, so we
    # keep it and ignore the deprecation warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
        existing_colors = plt.cm.get_cmap('tab10', len(unique_poolings))

    color_mapping = {}
    # creating a dictionnary that maps each value of pooling to a color
    for i, value in enumerate(unique_poolings):
        color_mapping[value] = existing_colors(i)
        # mapping the values through the dictionnary

    colors = np.array([color_mapping[value] for value in poolings])

    return poolings,color_mapping, colors



def get_convolution_mapping(list_dict : List[Dict]) -> Tuple[List, Dict, List] :
  """
  For the shape of the points of the following plot functions, we will
  need a mapping of the value of the architecture to the colors

  To do so, this function return the mapping (a dictionnary) and
  the list of all shapes corresponding to the architecture
  """
   
  # the convolutions will be used for the shape of the points
  convolutions = [d['convolution_layer'] for d in list_dict]

  if len(convolutions) > 41 :
    raise Exception(f"Not enough possible values of shape for the convolutions : got {len(convolutions)} but expected 41 at most")
  # same as for the colors: creating a mapping of the convolutions to the shapes
  unique_convolutions = np.unique(convolutions)
  existing_shapes = list(markers.MarkerStyle.markers.keys())
  shape_mapping = {}

  for i, value in enumerate(unique_convolutions):
    shape_mapping[value] = existing_shapes[i]

  shapes = [shape_mapping[value] for value in convolutions]

  return convolutions, shape_mapping, shapes



def plot_from_dict(list_dict : List[Dict], figsize : Tuple[int,int], **kwargs) -> None :
  """
  Plot the graph resulting from the list of dictionnary

  **kwargs -> additional keyword arguments passed to matplotlib functions

  Raises an error if the desired keys are not present in a dictionnary
  """
  # checking that there is no problem of keys in each given dictionnary
  key_values_to_check = ['nb_parameters',
                         'mean_accuracy',
                         'homophily',
                         'pooling',
                         'convolution_layer',
                         'dataset']

  for i,d in enumerate(list_dict) :

    if not(all(key in d for key in key_values_to_check)) :

      raise Exception(f"Problem of key for the {i}-th dictionnary")


  # the x,y and z of the scatter in 3D
  x = np.array([d['nb_parameters'] for d in list_dict])
  y = np.array([d['mean_accuracy'] for d in list_dict])
  z = np.array([d['homophily'] for d in list_dict])


  _, color_mapping, colors = get_pooling_mapping(list_dict)

  _, shape_mapping, shapes = get_convolution_mapping(list_dict)


  # scatter in 3D
  fig = plt.figure(figsize=figsize)
  plt.tight_layout()
  ax = ax = fig.add_subplot(111, projection='3d')

  # the legend for the colors
  colors_keys = list(color_mapping.keys())
  color_values = list(color_mapping.values())
  legend_color = [mpatches.Patch(color=color_values[i], label=f"{colors_keys[i]}-pool") for i in range(len(colors_keys))]

  # the legend for the shapes
  # To do so, a trick: plotting invisible lines for each shape
  shape_keys = list(shape_mapping.keys())
  shape_values = list(shape_mapping.values())
  legend_shape = [ax.scatter([], [], color='black', marker=shape_values[i], label=shape_keys[i]) for i in range(len(shape_keys))]


  for x_value, y_value, z_value, color, shape in zip(x, y, z, colors, shapes) :
    ax.scatter(x_value, y_value, z_value, c=color.reshape((1,4)), marker=shape, **kwargs)

  ax.legend(handles= legend_color + legend_shape)

  ax.set_xlabel("Number of parameters")
  ax.set_ylabel("Mean accuracy")
  ax.set_zlabel("Homophily", labelpad=0.)

  name_dataset = list_dict[0]['dataset']
  plt.savefig(f"./Visualisation/results/scatter_plot_3D-{name_dataset}.png")

  plt.show()

  return



def pairplot_from_dict(list_dict : List[Dict], 
                       rows_to_plot : List[Tuple[str,str]],
                       dim_grid_subplots : Tuple[int,int],
                       figsize : Tuple[int,int] | None = None,
                       **kwargs) -> None :
        
    '''
    Plot all the in subfigures all the variables described in rows_to_plot

    rows_to_plot -> All the variables of the dictionnaries of list_dict we want to
    plot (for instance, denoting (x,y) the first entry of rows_to_plot, the first plot will
    be dict[x] for the x-axis and dict[y] for the y-axis, for dict in list_dict)

    dim_grid_subplots -> the dimension of the grid of subplots (for instance (2,3) means that there
    are two rows and 3 columns)

    **kwargs -> additional keyword arguments passed to matplotlib functions

    Raises error if some elements of the rows don't correspond to key values in list_dict,
    or if the number of elements of rows_to_plot don't fit the dimension of dim_grid_subplots
    '''
    # the list of all keys present in list_dict
    keys_list_dict = [dic.keys() for dic in list_dict]

    for i, (key1, key2) in enumerate(rows_to_plot) :
        
        # check if key1 is present in the keys of all dictionnary
        if not(all(key1 in key for key in keys_list_dict)) :

            raise Exception(
                            f"The first key of the {i+1}-th element of the variables "
                            "to plot is not present in all keys of the dictionnaries "
                            )
        
        # same for key2
        if not(all(key2 in key for key in keys_list_dict)) :

            raise Exception(
                            f"The first key of the {i+1}-th element of the variables "
                             "to plot is not present in all keys of the dictionnaries "
                            )
        

    n_plot_rows, n_plot_cols = dim_grid_subplots

    if n_plot_rows*n_plot_cols != len(rows_to_plot) :

        raise Exception(
                        f"The number of plots imposed by the dimension of the grid "
                        f"({n_plot_rows} x {n_plot_cols} = {n_plot_rows*n_plot_cols}) "
                        f"is not consistent with the number of plots ({len(rows_to_plot)}) ")
    

    _, color_mapping, colors = get_pooling_mapping(list_dict)
    _, shape_mapping, shapes = get_convolution_mapping(list_dict)

    colors_keys = list(color_mapping.keys())
    color_values = list(color_mapping.values())

    shape_keys = list(shape_mapping.keys())
    shape_values = list(shape_mapping.values())


    _, axs = plt.subplots(n_plot_rows, n_plot_cols, figsize=figsize)

    axs = axs.flatten()

    n_dict = len(list_dict)


    for ax, (key1, key2) in zip(axs, rows_to_plot) :
        
        legend_color = [mpatches.Patch(color=color_values[i], label=f"{colors_keys[i]}-pool") for i in range(len(colors_keys))]
        legend_shape = [ax.scatter([], [], color='black', marker=shape_values[i], label=shape_keys[i]) for i in range(len(shape_keys))]

        x_values = [list_dict[i][key1] for i in range(n_dict)]
        y_values = [list_dict[i][key2] for i in range(n_dict)]

        for x_value, y_value, color, shape in zip(x_values, y_values, colors, shapes) :
          ax.scatter(x_value, y_value, c=color.reshape((1,4)), marker=shape, **kwargs)

        ax.legend(handles= legend_color + legend_shape)

        ax.set_xlabel(key1)
        ax.set_ylabel(key2)


    plt.tight_layout()

    name_dataset = list_dict[0]['dataset']
    plt.savefig(f"./Visualisation/results/pairplot-{name_dataset}.png")

    # Show the plot
    plt.show()

    return 




def to_table(list_dict : List[Dict]) -> pd.DataFrame:
  """
  Convert a list of dictionaries into a formatted Pandas DataFrame for tabular presentation.
  
  Parameters:
      list_dict (List[Dict]): A list of dictionaries containing the following keys:
          - 'dataset': The name of the dataset.
          - 'global_pooling_layer': The type of global pooling layer used.
          - 'local_pooling_layer': The type of local pooling layer used.
          - 'mean_accuracy': The mean accuracy value.
          - 'std_accuracy': The standard deviation of accuracy.

  Returns:
      pd.DataFrame: A formatted Pandas DataFrame with columns 'dataset', 'pooling_layer', and 'accuracy'.
          - 'dataset': Name of the dataset.
          - 'pooling_layer': Concatenation of 'local_pooling_layer' and 'global_pooling_layer'.
          - 'accuracy': Formatted string of mean accuracy ± standard deviation.

  Example Usage:
      list_dict = [
          {"dataset": "Dataset A", "global_pooling_layer": "Avg", "local_pooling_layer": "Max", "mean_accuracy": 0.85, "std_accuracy": 0.03},
          {"dataset": "Dataset B", "global_pooling_layer": "Sum", "local_pooling_layer": "Avg", "mean_accuracy": 0.92, "std_accuracy": 0.02}
      ]
      df = to_table(list_dict)
      print(df)
  
  """
  dic_results = {"dataset": [], "global_pooling_layer":[], "local_pooling_layer":[], "mean_accuracy":[], "std_accuracy":[]}

  for dic in list_dict:
      dic_results["dataset"].append(dic["dataset"])
      dic_results["global_pooling_layer"].append(dic["global_pooling_layer"])
      dic_results["local_pooling_layer"].append(dic["local_pooling_layer"])
      dic_results["mean_accuracy"].append(dic["mean_accuracy"])
      dic_results["std_accuracy"].append(dic["std_accuracy"])

  df = pd.DataFrame(dic_results)

  df["pooling_layer"] = df["local_pooling_layer"].astype(str).replace("None", "") + df["global_pooling_layer"]
  df["accuracy"] = "$" + df["mean_accuracy"].apply("{:.3f}".format).astype(str) + "\pm" + df["std_accuracy"].apply("{:.3f}".format).astype(str) + "$"
  df = df.drop(columns=["local_pooling_layer", "global_pooling_layer", "mean_accuracy", "std_accuracy"])
  df = df.pivot(index='dataset', columns='pooling_layer', values='accuracy')
  df = df.rename_axis(None, axis=1)
  df = df.rename_axis(None, axis=0)

  return df

def plot_losses(list_dict, train="train"):
    for dict in list_dict:
        if 'split 1' in dict:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(np.arange(len(dict['split 1']['train_losses'])), dict['split 1'][train + '_losses'])
            ax.set_xlabel("Epochs")
            ax.set_ylabel(train + " loss")
            if dict["local_pooling_layer"] is not None:
                ax.set_title(train + " loss across epochs with early stopping" +  "\n" +
                            "for " + dict["dataset"] + " with " + dict['convolution_layer'] + 
                             "," + dict["local_pooling_layer"] + " and " + dict["global_pooling_layer"])
                plt.savefig("./Visualisation/results/losses/" + train + "/" + dict["dataset"] +
                            "_"  + dict['convolution_layer'] + "_" + dict["local_pooling_layer"] +
                            "_" + dict["global_pooling_layer"] + ".png")
                plt.close()
            else:
                ax.set_title(train + " loss across epochs with early stopping" +  "\n" +
                            "for " + dict["dataset"] + " with " + dict['convolution_layer'] +
                             "," + "None " + " and " +  dict["global_pooling_layer"])
                plt.savefig("./Visualisation/results/losses/" + train + "/" + dict["dataset"] +
                            "_" +dict['convolution_layer'] + "_" +  "None_" +
                            dict["global_pooling_layer"] + ".png")
                plt.close()

def plot_acc_parameters(list_dict):
    datasets = set([dict['dataset'] for dict in list_dict])
    for dataset in datasets:
        plt.figure(figsize=(8, 6))
        color_map = 'viridis'
        color_index = 0
        nb_parameters = []
        mean_accuracy = []
        for split in list_dict:
            if split['dataset'] == dataset:
                nb_parameters.append(split['nb_parameters'])
                mean_accuracy.append(split['mean_accuracy'])
                if split['local_pooling_layer'] is not None:
                    plt.scatter(split['nb_parameters'], split['mean_accuracy'], 
                                label=split['convolution_layer'] + "_" + split['global_pooling_layer'] +"_"+ split['local_pooling_layer'], 
                                cmap="viridis")
                else:
                    plt.scatter(split['nb_parameters'], split['mean_accuracy'], 
                                label=split['convolution_layer'] + "_" + split['global_pooling_layer'] +"_None", cmap="viridis")
        plt.xlabel('Number of Parameters')
        plt.ylabel('Mean Accuracy')
        plt.title(f'Mean Accuracy vs Number of Parameters with Different Pooling Layers for {dataset}')
        plt.legend(loc='lower right', bbox_to_anchor=(1.3, 0), borderaxespad=0.)
        plt.savefig("./Visualisation/results/acc_parameters/" + dataset + ".png", bbox_inches='tight')
        plt.show()

def plot_acc(list_dict, train="train"):
    for dict in list_dict:
        if 'split 1' in dict:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(np.arange(len(list_dict[0]['split 1']['train_accuracies'])), list_dict[0]['split 1'][train + '_accuracies'])
            ax.set_xlabel("Epochs")
            ax.set_ylabel(train + " acc")
            if dict["local_pooling_layer"] is not None:
                ax.set_title(train + " accuracy across epochs with early stopping" +  "\n" +
                            "for " + dict["dataset"] + " with " + dict['convolution_layer'] + 
                             "," + dict["local_pooling_layer"] + " and " + dict["global_pooling_layer"])
                plt.savefig("./Visualisation/results/acc/" + train + "/" + dict["dataset"] +
                            "_"  + dict['convolution_layer'] + "_" + dict["local_pooling_layer"] +
                            "_" + dict["global_pooling_layer"] + ".png")
                plt.close()
            else:
                ax.set_title(train + " accuracy across epochs with early stopping" +  "\n" +
                            "for " + dict["dataset"] + " with " + dict['convolution_layer'] +
                             "," + "None " + " and " +  dict["global_pooling_layer"])
                plt.savefig("./Visualisation/results/acc/" + train + "/" + dict["dataset"] +
                            "_" +dict['convolution_layer'] + "_" +  "None_" +
                            dict["global_pooling_layer"] + ".png")
                plt.close()