import yaml
import os
from training_script import train_model_from_config

global_pooling_layer_to_test = ["mean", "max"]
local_pooling_layers_to_test = ["SAG","MEWIS", None]
convolution_layers_to_test = ["GCN", "GAT"]
# Define your configuration data

path_templates = "configs/templates"
path_generated = "configs/generated"
configs_path = os.listdir(path_templates)

for config_path in configs_path:
    if config_path.endswith(".yml"):
        with open(os.path.join(path_templates,config_path), 'r') as config_file:
            config_model = yaml.safe_load(config_file)
    
        dataset_name = config_model["model"]["dataset"]
        for convolution_layer in convolution_layers_to_test:
            config_model["model"]["convolution_layer"] = convolution_layer
            for global_pooling_layer in global_pooling_layer_to_test:
                config_model["model"]["global_pooling_layer"] = global_pooling_layer
                for local_pooling_layer in local_pooling_layers_to_test:
                    config_model["model"]["local_pooling_layer"] = local_pooling_layer
                    # Specify the file path where you want to save the YAML configuration
                    config_name = f"{dataset_name}_{convolution_layer}_{global_pooling_layer}_{local_pooling_layer}.yaml"
    
                    # Write the configuration data to the YAML file
                    with open(os.path.join(path_generated,config_name), 'w') as config_file:
                        yaml.dump(config_model, config_file)


configs_generated_path = os.listdir(path_generated)

for config_path in configs_generated_path:
    if config_path.endswith(".yaml"):
        train_model_from_config(os.path.join(path_generated,config_path))
    