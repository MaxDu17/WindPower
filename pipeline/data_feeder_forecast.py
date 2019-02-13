import pandas as pd

class DataParser_Forecast:
    """
    this database has a query interval of [x, y)
    meaning that xth number is included, while the yth number is not

    This is the lowest level library; everything else depends on this.
    """
    def __init__(self):
        self.data = pd.read_csv("../Training_Sets/ALL_DATA_NORMALIZED.csv")  # read file
        self.data.drop(["Minutes"], axis = 1, inplace=True)
        self.combined_data =  self.data

    def use_foreign(self, file_name):
        self.data = pd.read_csv(file_name)  # read file
        self.data.drop(["Minutes"], axis = 1, inplace=True)
        self.combined_data =  self.data

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

