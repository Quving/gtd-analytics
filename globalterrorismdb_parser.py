import json
import os

import pandas as pd


class GlobalTerrorismDBParser:
    def __init__(self):
        """ Reads in the csv file and stores it as DataFrame object. """
        with open("country_dict.json") as f:
            self.country_dict = json.load(f)

        self.data_dir = "data"
        self.data_filename = "globalterrorismdb_0617dist.csv"
        self.data_path = os.path.join(self.data_dir, self.data_filename)

        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Must be instance of DataFrame, but got " + type(self.data) + ".")

    def reset_jsons(self):
        """
        Replaces the 'column.jsons' with the original file.
        :return:
        """
        self.data = pd.read_csv(self.data_path,
                                encoding="latin1",
                                low_memory=False)
        self.to_json()

    def get_country_by_id(self, id):
        """
        Returns the country as label by a given id.
        :param id: str, int
        :return:
        """
        if not str(id) in self.country_dict:
            return "Undefined"

        return self.country_dict[str(id)]

    def to_json(self):
        """ Stores a json with the content of each column in the csv file. """

        for column in self.data.columns:
            with open(os.path.join(self.data_dir, column + ".json"), "w") as f:
                json.dump(obj=list(self.data.get(column)),
                          fp=f,
                          sort_keys=True,
                          indent=4)

    def get_column(self, column):
        """
        Returns from the data-set a column as list.
        :param column:
        :return:
        """
        filename = os.path.join(self.data_dir, column + ".json")

        if not os.path.exists(filename):
            raise KeyError("'" + column + "' does not exist.")
        with open(filename, "r") as f:
            out = json.loads(column)

        return list(out)
