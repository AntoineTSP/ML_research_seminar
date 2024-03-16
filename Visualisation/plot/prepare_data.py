from typing import List, Dict

from json import load
import os
from mimetypes import guess_type

import pandas as pd


class Prepare :

    def __init__(self) :

        self._path_list_dict = os.path.join("..", "model", "results")
        self._path_homophily = os.path.join("..", "homophily", "homophily_data.csv")

        self.set_list_dict()
        self.set_cross_dataframe()


    def set_list_dict(self) :
        """
        Return the list of dictionary that is used for most
        visualisation task
        """
        list_dict = []

        with os.scandir(self._path_list_dict) as entries:
            for entry in entries:
                if entry.is_file():
                    name = entry.name
                    if 'json' in guess_type(name)[0]:
                        # Construct the full path using os.path.join again
                        full_path = os.path.join(self._path_list_dict, name)
                        with open(full_path, 'r') as file:
                            data = load(file)
                        list_dict.append(data)

        self.list_dict = list_dict


    def set_cross_dataframe(self) :

        df_homophily = pd.read_csv(self._path_homophily)
        df_homophily['Name_Dataset'] = df_homophily['Name_Dataset'].apply(lambda s : s.upper())

        for dic in self.list_dict :

            name_dataset = dic["dataset"]
            dic["homophily"] = df_homophily.loc[df_homophily['Name_Dataset'] == name_dataset, 'Homophily_edge_train'].values[0]

            if dic['local_pooling_layer'] is None:
                dic['local_pooling_layer'] = "None"

            dic["pooling_and_archi"] = f"{dic['local_pooling_layer']}+{dic['convolution_layer']}"


    def get_list_dict(self) -> List[Dict] :

        return self.list_dict


def get_list_dict() -> List[Dict] :
    prepare = Prepare()
    return prepare.get_list_dict()