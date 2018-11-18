import pandas as pd
import numpy as np
class DataParser_Weather:
    """
    this database has a query interval of [x, y)
    meaning that xth number is included, while the yth number is not

    This is the lowest level library; everything else depends on this.
    """
    def __init__(self):
        #self.data = pd.read_csv("Training_sets/83863-2011.csv", skiprows = 3) #read file
        self.data = pd.read_csv("C:/Users/Max Du/Desktop/Repos/CSIRE/Training_sets/2012DATA_NORMALIZED.csv")  # read file
        #clean_data = data[["Month", "Day", "Hour", "Minute", "power (MW)"]] #extract critical data, not used here
        self.combined_data = self.data[["native_power", "power", "wind_dir", "wind_speed", "air_temp", "SAP", "air_density"]] #extracts ALL data


    def use_foreign(self, file_name):
        self.data = pd.read_csv(file_name) #read file
        #clean_data = data[["Month", "Day", "Hour", "Minute", "power (MW)"]] #extract critical data, not used here
        self.combined_data = self.data[["power", "wind_dir", "wind_speed", "air_temp", "SAP", "air_density"]]  # extracts ALL data

    def print_from_start(self, number):
        return self.combined_data.head(number) #print everything. Seldom used, but is an option

    def dataset_size(self):
        return len(self.combined_data)

    def grab_list_range(self,start,end): #selects a range to query
        self.combined_data.index.name = "index" #sets index to "index" for ease of query
        command = str(start) + "<=index<" + str(end) #makes command
        subset = self.combined_data.query(command) #querys the pandas data frame
        return subset.values.tolist()  #returns the query in a form of a list

    def grab_element(self, element):
        array = self.combined_data.values
        clean_array = [k[0] for k in array]
        return clean_array[element]

