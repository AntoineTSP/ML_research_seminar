from typing import List, Dict, Tuple, Any

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.markers as markers
import matplotlib.patches as mpatches

import pandas as pd


def get_convolution_mapping(list_dict: List[Dict]) -> Tuple[List, Dict, List]:
    """
    For the shape of the points of the following plot functions, we will
    need a mapping of the value of the architecture to the shapes

    To do so, this function return the mapping (a dictionnary) and
    the list of all shapes corresponding to the architecture
    """

    # the convolutions will be used for the shape of the points
    convolutions = [d["convolution_layer"] for d in list_dict]
    unique_convolutions = np.unique(convolutions)

    if len(unique_convolutions) > 41:
        raise Exception(
            f"Not enough possible values of shape for the convolutions : got {len(convolutions)} but expected 41 at most"
        )
    # same as for the colors: creating a mapping of the convolutions to the shapes
    
    existing_shapes = list(markers.MarkerStyle.markers.keys())
    shape_mapping = {}

    for i, value in enumerate(unique_convolutions):
        shape_mapping[value] = existing_shapes[i]

    shapes = [shape_mapping[value] for value in convolutions]

    return convolutions, shape_mapping, shapes


def get_pooling_mapping(list_dict: List[Dict]) -> Tuple[List, Dict, List]:
    """
    For the color of the points of the following plot functions, we will
    need a mapping of the pooling to the colors

    To do so, this function return the mapping (a dictionnary) and
    the list of all colors corresponding to the pooling
    """
    poolings = [d["pooling"] for d in list_dict]
    unique_poolings = np.unique(poolings)

    existing_colors = plt.get_cmap("tab10", len(unique_poolings))

    color_mapping = {}
    # creating a dictionnary that maps each value of pooling to a color

    for i, value in enumerate(unique_poolings):
        color_mapping[value] = existing_colors(i)

    # mapping the values through the dictionnary
    colors = [color_mapping[value] for value in poolings]

    return poolings, color_mapping, colors


def plot_from_dict(list_dict: List[Dict], figsize: Tuple[int, int] = (10,6), **kwargs) -> None:
    """
    Plot the graph resulting from the list of dictionnary

    **kwargs -> additional keyword arguments passed to matplotlib functions

    Raises an error if the desired keys are not present in a dictionnary
    """
    # checking that there is no problem of keys in each given dictionnary
    key_values_to_check = [
        "nb_parameters",
        "mean_accuracy",
        "homophily",
        "pooling",
        "convolution_layer",
        "dataset",
    ]

    for i, d in enumerate(list_dict):

        if not (all(key in d for key in key_values_to_check)):

            raise Exception(f"Problem of key for the {i}-th dictionnary")

    # the x,y and z of the scatter in 3D
    x = np.array([d["nb_parameters"] for d in list_dict])
    y = np.array([d["mean_accuracy"] for d in list_dict])
    z = np.array([d["homophily"] for d in list_dict])

    _, color_mapping, colors = get_pooling_mapping(list_dict)

    _, shape_mapping, shapes = get_convolution_mapping(list_dict)

    # scatter in 3D
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = ax = fig.add_subplot(111, projection="3d")

    # the legend for the colors
    colors_keys = list(color_mapping.keys())
    color_values = list(color_mapping.values())
    legend_color = [
        mpatches.Patch(color=color_values[i], label=f"{colors_keys[i]}-pool")
        for i in range(len(colors_keys))
    ]

    # the legend for the shapes
    # To do so, a trick: plotting invisible lines for each shape
    shape_keys = list(shape_mapping.keys())
    shape_values = list(shape_mapping.values())
    legend_shape = [
        ax.scatter([], [], color="black", marker=shape_values[i], label=shape_keys[i])
        for i in range(len(shape_keys))
    ]

    for x_value, y_value, z_value, color, shape in zip(x, y, z, colors, shapes):
        ax.scatter(
            x_value,
            y_value,
            z_value,
            c=np.array(color).reshape((1, 4)),
            marker=shape,
            **kwargs,
        )

    ax.legend(
            handles=legend_color + legend_shape,
            bbox_to_anchor=(1.5,1)
        )

    ax.set_xlabel("Number of parameters")
    ax.set_ylabel("Mean accuracy")
    ax.set_zlabel("Homophily", labelpad=0.0)

    plt.savefig("./Visualisation/results/scatter_plot_3D.png")

    plt.show()

    return


def pairplot_from_dict(
    list_dict: List[Dict],
    rows_to_plot: List[Tuple[str, str]],
    dim_grid_subplots: Tuple[int, int],
    figsize: Tuple[int, int] | None = None,
    kwargs1 : Dict[str, Any] = {},
    kwargs2 : Dict[str, Any] = {},
) -> None:
    """
    Plot all the in subfigures all the variables described in rows_to_plot

    rows_to_plot -> All the variables of the dictionnaries of list_dict we want to
    plot (for instance, denoting (x,y) the first entry of rows_to_plot, the first plot will
    be dict[x] for the x-axis and dict[y] for the y-axis, for dict in list_dict)

    dim_grid_subplots -> the dimension of the grid of subplots (for instance (2,3) means that there
    are two rows and 3 columns)

    **kwargs -> additional keyword arguments passed to matplotlib functions

    Raises error if some elements of the rows don't correspond to key values in list_dict,
    or if the number of elements of rows_to_plot don't fit the dimension of dim_grid_subplots
    """
    # the list of all keys present in list_dict
    keys_list_dict = [dic.keys() for dic in list_dict]

    for i, (key1, key2) in enumerate(rows_to_plot):

        # check if key1 is present in the keys of all dictionnary
        if not (all(key1 in key for key in keys_list_dict)):

            raise Exception(
                f"The first key of the {i+1}-th element of the variables "
                "to plot is not present in all keys of the dictionnaries "
            )

        # same for key2
        if not (all(key2 in key for key in keys_list_dict)):

            raise Exception(
                f"The first key of the {i+1}-th element of the variables "
                "to plot is not present in all keys of the dictionnaries "
            )

    n_plot_rows, n_plot_cols = dim_grid_subplots

    if n_plot_rows * n_plot_cols != len(rows_to_plot):

        raise Exception(
            f"The number of plots imposed by the dimension of the grid "
            f"({n_plot_rows} x {n_plot_cols} = {n_plot_rows*n_plot_cols}) "
            f"is not consistent with the number of plots ({len(rows_to_plot)}) "
        )

    _, color_mapping, colors = get_pooling_mapping(list_dict)
    _, shape_mapping, shapes = get_convolution_mapping(list_dict)

    colors_keys = list(color_mapping.keys())
    color_values = list(color_mapping.values())

    shape_keys = list(shape_mapping.keys())
    shape_values = list(shape_mapping.values())

    _, axs = plt.subplots(n_plot_rows, n_plot_cols, figsize=figsize)

    axs = axs.flatten()

    n_dict = len(list_dict)

    for ax, (key1, key2) in zip(axs, rows_to_plot):

        legend_color = [
            mpatches.Patch(color=color_values[i], label=f"{colors_keys[i]}-pool")
            for i in range(len(colors_keys))
        ]
        legend_shape = [
            ax.scatter(
                [], [], color="black", marker=shape_values[i], label=shape_keys[i]
            )
            for i in range(len(shape_keys))
        ]

        x_values = [list_dict[i][key1] for i in range(n_dict)]
        y_values = [list_dict[i][key2] for i in range(n_dict)]

        for x_value, y_value, color, shape in zip(x_values, y_values, colors, shapes):
            ax.scatter(
                x_value,
                y_value,
                c=np.array(color).reshape((1, 4)),
                marker=shape,
                **kwargs1,
            )

        ax.legend(handles=legend_color + legend_shape, **kwargs2)

        ax.set_xlabel(key1)
        ax.set_ylabel(key2)

    plt.tight_layout()

    plt.savefig("./Visualisation/results/pairplot.png")

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
  dic_results = {"dataset": [], "global_pooling_layer":[], "local_pooling_layer":[], "convolution_layer":[], "mean_accuracy":[], "std_accuracy":[], "training_time":[]}

  for dic in list_dict:
        dic_results["dataset"].append(dic["dataset"])
        dic_results["global_pooling_layer"].append(dic["global_pooling_layer"])
        dic_results["local_pooling_layer"].append(dic["local_pooling_layer"])
        dic_results["convolution_layer"].append(dic["convolution_layer"])
        dic_results["mean_accuracy"].append(dic["mean_accuracy"])
        dic_results["std_accuracy"].append(dic["std_accuracy"])
        training_time = 0
        for i in range(1, 11):
            try:
                training_time += dic["split "+str(i)]["train_time_per_epoch"]*dic["split "+str(i)]["last_epoch"]
            except KeyError:
                training_time += 0
        dic_results["training_time"].append(training_time/10)

  df = pd.DataFrame(dic_results)

  df["local_pooling_layer"] = df["local_pooling_layer"].astype(str)
  indexes_to_bold = df.groupby("dataset")["mean_accuracy"].idxmax()
  df["accuracy"] = "$" + df["mean_accuracy"].apply("{:.3f}".format).astype(str) + "\pm" + df["std_accuracy"].apply("{:.3f}".format).astype(str) + "$"
  df.loc[indexes_to_bold, "accuracy"] = "$\\bm{" + df.loc[indexes_to_bold, "mean_accuracy"].apply("{:.3f}".format).astype(str) + "\pm" + df.loc[indexes_to_bold, "std_accuracy"].apply("{:.3f}".format).astype(str) + "}$"
  df = df.drop(columns=["mean_accuracy", "std_accuracy"])
  df = df.rename(columns={'convolution_layer': 'Conv', 'local_pooling_layer': 'Local', 'global_pooling_layer': 'Global', 'dataset': 'Dataset'})
  df = df.pivot(index=['Conv', "Local", 'Global'], columns='Dataset', values=['accuracy', 'training_time'])
  training_time = df["training_time"].mean(axis=1).copy().astype(int).astype(str)
  df = df.drop(columns=["training_time"])
  df.columns = df.columns.droplevel(0)
  df = df.rename_axis(None, axis=1)  
  df["Training Time"] = training_time
  return df

def plot_bar_dataset(
    list_dict: List[Dict], cmap: str = "tab10", n_colors: int = 10, **kwargs
):

    # first, create a dictionnary whose keys are the dataset and values are
    # the list of all element from list_dict for this dataset
    data_by_dataset = {}

    for entry in list_dict:

        dataset = entry["dataset"]

        if dataset not in data_by_dataset:
            data_by_dataset[dataset] = []

        data_by_dataset[dataset].append(entry)

    # the colors of the bars
    colors = [plt.get_cmap(cmap)(i) for i in range(n_colors)]

    _, axes = plt.subplots(
        nrows=len(data_by_dataset),
        ncols=1,
        figsize=(12, len(data_by_dataset) * 5),
        squeeze=False,
    )
    axes = axes.flatten()

    for ax, (dataset, records) in zip(axes, data_by_dataset.items()):

        # all different pooling methods
        pooling_methods = np.unique([record["pooling"] for record in records])
        mean_accuracies = {pol: [] for pol in pooling_methods}

        for record in records:
            pol = record["pooling"]
            mean_accuracies[pol].append(record["mean_accuracy"])

        # the list of the mean accuracy for each pooling value (averaged per pooling)
        mean_accuracies = list(map(lambda l: sum(l) / len(l), mean_accuracies.values()))

        # repeating colors if there is more pooling methods than pooling
        if len(pooling_methods) > n_colors:
            colors = colors * (len(pooling_methods) // n_colors + 1)

        bars = ax.bar(
            pooling_methods,
            mean_accuracies,
            color=colors[: len(pooling_methods)],
            **kwargs,
        )
        ax.set_title(f"Dataset: {dataset}")
        ax.set_ylabel("Mean Accuracy")
        ax.set_xlabel("Pooling Method")
        ax.set_xticks(range(len(pooling_methods)))

        for bar, acc in zip(bars, mean_accuracies):
            height = bar.get_height()
            ax.annotate(
                f"{acc:.3f}",
                (bar.get_x() + bar.get_width() / 2, height / 2),
                textcoords="offset points",
                xytext=(0, 0),
                ha="center",
                va="center",
                rotation=0,
            )

    # Adjust the spacing after creating subplots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)

    plt.show()

def plot_losses(list_dict, train="train"):
    for dict in list_dict:
        if "split 1" in dict:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(
                np.arange(len(dict["split 1"]["train_losses"])),
                dict["split 1"][train + "_losses"],
            )
            ax.set_xlabel("Epochs")
            ax.set_ylabel(train + " loss")
            if dict["local_pooling_layer"] is not None:
                ax.set_title(
                    train
                    + " loss across epochs with early stopping"
                    + "\n"
                    + "for "
                    + dict["dataset"]
                    + " with "
                    + dict["convolution_layer"]
                    + ","
                    + dict["local_pooling_layer"]
                    + " and "
                    + dict["global_pooling_layer"]
                )
                plt.savefig(
                    "./Visualisation/results/losses/"
                    + train
                    + "/"
                    + dict["dataset"]
                    + "_"
                    + dict["convolution_layer"]
                    + "_"
                    + dict["local_pooling_layer"]
                    + "_"
                    + dict["global_pooling_layer"]
                    + ".png"
                )
                plt.close()
            else:
                ax.set_title(
                    train
                    + " loss across epochs with early stopping"
                    + "\n"
                    + "for "
                    + dict["dataset"]
                    + " with "
                    + dict["convolution_layer"]
                    + ","
                    + "None "
                    + " and "
                    + dict["global_pooling_layer"]
                )
                plt.savefig(
                    "./Visualisation/results/losses/"
                    + train
                    + "/"
                    + dict["dataset"]
                    + "_"
                    + dict["convolution_layer"]
                    + "_"
                    + "None_"
                    + dict["global_pooling_layer"]
                    + ".png"
                )
                plt.close()


def plot_acc_parameters(list_dict):
    if not list_dict:
        print("Error: Empty list")
        return

    datasets = set(d['dataset'] for d in list_dict)
    for dataset in datasets:
        plt.figure(figsize=(8, 6))
        colors = plt.cm.hsv(np.linspace(0, 1, 100))  # Define 15 different colors
        color_map = plt.cm.colors.ListedColormap(colors)  # Create a custom colormap
        nb_parameters = []
        mean_accuracy = []
        labels = set()
        for split in list_dict:
            if split.get('dataset') == dataset:
                nb_parameters.append(split.get('nb_parameters', 0))
                mean_accuracy.append(split.get('mean_accuracy', 0))
                label = f"{split.get('convolution_layer')}_{split.get('local_pooling_layer', 'None')}"
                labels.add(label)

        sorted_labels = sorted(labels)  # Sort labels alphabetically
        
        colors = [color_map(i / len(sorted_labels)) for i in range(len(sorted_labels))]
        label_color_dict = {label: color for label, color in zip(sorted_labels, colors)}
        
        for split in list_dict:
            if split.get('dataset') == dataset:
                label = f"{split.get('convolution_layer')}_{split.get('local_pooling_layer', 'None')}"
                color = label_color_dict.get(label, 'black')  # Use black color for labels not in sorted_labels
                # split.get('global_pooling_layer')
                # print(split.get("global_pooling_layer"))
                if split.get("global_pooling_layer") == "mean":
                    plt.scatter(split.get('nb_parameters', 0), split.get('mean_accuracy', 0),
                                label=label if label in sorted_labels else None, color=color, marker='^')
                if split.get("global_pooling_layer") == "max":
                    plt.scatter(split.get('nb_parameters', 0), split.get('mean_accuracy', 0),
                                label=label if label in sorted_labels else None, color=color, marker='o')

        plt.xlabel('Number of Parameters')
        plt.ylabel('Mean Accuracy')
        plt.title(f'Mean Accuracy vs Number of Parameters with Different Pooling Layers for {dataset} \n Triangle = mean ending pooling | Circle = max ending pooling')
        plt.legend(loc='lower right', bbox_to_anchor=(1.5, 0), borderaxespad=0., prop={'size': 10})
        plt.savefig(f"./Visualisation/results/acc_parameters/{dataset}.png", bbox_inches='tight')
        plt.close()

def plot_acc_time_epoch(list_dict):
    if not list_dict:
        print("Error: Empty list")
        return

    datasets = set(d['dataset'] for d in list_dict)
    for dataset in datasets:
        plt.figure(figsize=(8, 6))
        colors = plt.cm.hsv(np.linspace(0, 1, 100))  # Define 15 different colors
        color_map = plt.cm.colors.ListedColormap(colors)  # Create a custom colormap
        train_time_per_epoch = []
        mean_accuracy = []
        labels = set()
        for split in list_dict:
            if 'split 1' in split:
                if split.get('dataset') == dataset:
                    train_time_per_epoch.append(split['split 1'].get('train_time_per_epoch', 0))
                    mean_accuracy.append(split.get('mean_accuracy', 0))
                    label = f"{split.get('convolution_layer')}_{split.get('local_pooling_layer', 'None')}"
                    labels.add(label)

        sorted_labels = sorted(labels)  # Sort labels alphabetically
        
        colors = [color_map(i / len(sorted_labels)) for i in range(len(sorted_labels))]
        label_color_dict = {label: color for label, color in zip(sorted_labels, colors)}
        
        for split in list_dict:
            if 'split 1' in split:
                if split.get('dataset') == dataset:
                    label = f"{split.get('convolution_layer')}_{split.get('local_pooling_layer', 'None')}"
                    color = label_color_dict.get(label, 'black')  # Use black color for labels not in sorted_labels
                    # split.get('global_pooling_layer')
                    # print(split.get("global_pooling_layer"))
                    if split.get("global_pooling_layer") == "mean":
                        plt.scatter(split['split 1'].get('train_time_per_epoch', 0), split.get('mean_accuracy', 0),
                                    label=label if label in sorted_labels else None, color=color, marker='^')
                    if split.get("global_pooling_layer") == "max":
                        plt.scatter(split['split 1'].get('train_time_per_epoch', 0), split.get('mean_accuracy', 0),
                                    label=label if label in sorted_labels else None, color=color, marker='o')

        plt.xlabel('Train time per epoch (s)')
        plt.ylabel('Mean Accuracy')
        plt.title(f'Mean Accuracy vs Train time per epoch with Different Pooling Layers for {dataset} \n Triangle = mean ending pooling | Circle = max ending pooling')
        plt.legend(loc='lower right', bbox_to_anchor=(1.5, 0), borderaxespad=0., prop={'size': 10})
        plt.savefig(f"./Visualisation/results/acc_train_time_per_epoch/{dataset}.png", bbox_inches='tight')
        plt.close()

def plot_acc_full_train_time(list_dict):
    if not list_dict:
        print("Error: Empty list")
        return

    datasets = set(d['dataset'] for d in list_dict)
    for dataset in datasets:
        plt.figure(figsize=(8, 6))
        colors = plt.cm.hsv(np.linspace(0, 1, 100))  # Define 15 different colors
        color_map = plt.cm.colors.ListedColormap(colors)  # Create a custom colormap
        train_time = []
        mean_accuracy = []
        labels = set()
        for split in list_dict:
            if 'split 1' in split:
                if split.get('dataset') == dataset:
                    train_time.append(split['split 1'].get('train_time_per_epoch', 0) * len(split['split 1']['train_losses']))
                    mean_accuracy.append(split.get('mean_accuracy', 0))
                    label = f"{split.get('convolution_layer')}_{split.get('local_pooling_layer', 'None')}"
                    labels.add(label)

        sorted_labels = sorted(labels)  # Sort labels alphabetically
        
        colors = [color_map(i / len(sorted_labels)) for i in range(len(sorted_labels))]
        label_color_dict = {label: color for label, color in zip(sorted_labels, colors)}
        
        for split in list_dict:
            if 'split 1' in split:
                if split.get('dataset') == dataset:
                    label = f"{split.get('convolution_layer')}_{split.get('local_pooling_layer', 'None')}"
                    color = label_color_dict.get(label, 'black')  # Use black color for labels not in sorted_labels
                    # split.get('global_pooling_layer')
                    # print(split.get("global_pooling_layer"))
                    if split.get("global_pooling_layer") == "mean":
                        plt.scatter(split['split 1'].get('train_time_per_epoch', 0) * len(split['split 1']['train_losses']) / 60,
                                    split.get('mean_accuracy', 0),
                                    label=label if label in sorted_labels else None, color=color, marker='^')
                    if split.get("global_pooling_layer") == "max":
                        plt.scatter(split['split 1'].get('train_time_per_epoch', 0) * len(split['split 1']['train_losses']) /60 ,
                                    split.get('mean_accuracy', 0),
                                    label=label if label in sorted_labels else None, color=color, marker='o')

        plt.xlabel('Full train time (min)')
        plt.ylabel('Mean Accuracy')
        plt.title(f'Mean Accuracy vs Full train time with Different Pooling Layers for {dataset} \n Triangle = mean ending pooling | Circle = max ending pooling')
        plt.legend(loc='lower right', bbox_to_anchor=(1.5, 0), borderaxespad=0., prop={'size': 10})
        plt.savefig(f"./Visualisation/results/acc_full_train_time/{dataset}.png", bbox_inches='tight')
        plt.close()


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


def plot_acc_and_loss(list_dict, train="train"):
    for dict in list_dict:
        if 'split 1' in dict:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            color = 'tab:blue'
            ax1.plot(np.arange(len(dict['split 1']['train_accuracies'])), dict['split 1'][train + '_accuracies'],
                    color=color, label= train + " accuracy")
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel(train + " acc", color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            # Creating a secondary y-axis for the second curve
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.plot(np.arange(len(dict['split 1'][train + '_losses'])), dict['split 1'][train + '_losses'],
                    color=color, label=train + " loss")
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel(train + " loss", color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            # Adding legends
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left')
            
            if dict["local_pooling_layer"] is not None:
                ax1.set_title(train + " accuracy and loss across epochs with early stopping" +  "\n" +
                            "for " + dict["dataset"] + " with " + dict['convolution_layer'] + 
                             "," + dict["local_pooling_layer"] + " and " + dict["global_pooling_layer"])
                plt.tight_layout()
                plt.savefig("./Visualisation/results/acc_and_loss/" + train + "/" + dict["dataset"] +
                            "_"  + dict['convolution_layer'] + "_" + dict["local_pooling_layer"] +
                            "_" + dict["global_pooling_layer"] + ".png")
                plt.close()
            else:
                ax1.set_title(train + " accuracy and loss across epochs with early stopping" +  "\n" +
                            "for " + dict["dataset"] + " with " + dict['convolution_layer'] +
                             "," + "None " + " and " +  dict["global_pooling_layer"])
                plt.tight_layout()
                plt.savefig("./Visualisation/results/acc_and_loss/" + train + "/" + dict["dataset"] +
                            "_" +dict['convolution_layer'] + "_" +  "None_" +
                            dict["global_pooling_layer"] + ".png")
                plt.close()