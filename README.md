# ML_research_seminar
This will contain all the file for the M2DS course : ML research seminar

# To use training_script

```
python training_script.py -c configs/config_test.yml
```

One has to properly fill the yml file. Don't forget to install the yaml module.

New layers of convolution or pooling have to be added to model\layer_selector.py.

New layers of local pooling should also contained a dictionary indicating the order of the variables returned by the forward of the pooling layer. For instance the MEWISPooling has the dictionary : 

```
{'node_features':0,'edge_index':1,'batch':2, 'loss':3}
```

While the SAGPooling has the dictionary :

```
{'node_features':0,'edge_index':1,'batch':3}
```

Since the forward method of SAGPooling returns an edge_attributes values at the second place (which is useless to us), and similarly the MEWISPooling returns a loss at the third place that has to be taken into account.

# To do list

- Replace those ugly prints by a logger
