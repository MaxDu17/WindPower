from data_feeder_v2 import DataParser_Weather
from hyperparameters import Hyperparameters

class SetMaker_Weather:
    def __init__(self): #initializing variables used in calculation
        self.dp = DataParser_Weather()
        self.hyp = Hyperparameters()
        self.master_list = list()
        self.counter = 0
        self.batch_counter = 0
        self.training_set_size = 0
        self.valid_counter = 0
        self.validation_set_size = 0
        self.self_prompt_counter = 0
        self.running_list = list()
        self.label_list = list()

    def use_foreign(self, file_name): #wrapper function used to load a different-than-normal dataset, for foreign testing
        self.dp.use_foreign(file_name)

    def create_training_set(self): #initializing statement for training
        self.training_set_size = self.hyp.TRAIN_PERCENT * self.dp.dataset_size()
        self.test_counter = self.training_set_size

    def create_validation_set(self): #initializing phrase for validation
        self.validation_set_size = int(self.hyp.VALIDATION_PERCENT * self.dp.dataset_size()) #just casting to whole #

    def next_epoch_waterfall(self): #this just returns the entire sequence. DO NOT CALL NEXT_SAMPLE WITH THIS. Only for training
        carrier = False
        self.master_list = list()
        if self.counter + self.hyp.FOOTPRINT+1 > self.training_set_size:
            self.clear_counter()
            carrier = True
            print("wraparound")
        self.master_list = self.dp.grab_list_range(self.counter, self.counter+self.hyp.FOOTPRINT+1)
        self.counter += self.hyp.FOOTPRINT
        self.batch_counter = 0
        input_data = [k[1:] for k in self.master_list[:-1]]
        return carrier, input_data

    def next_epoch_test_waterfall(self): #this is nextepochtestsingleshift but with a waterfall
        if self.test_counter == 0:
            raise Exception("you forgot to initialize the test_counter! Execute create_training_set")
        if self.test_counter + self.hyp.FOOTPRINT + 1 > self.dp.dataset_size():
            raise ValueError("you have reached the end of the test set. Violation dataset_maker/next_epoch_test")
        self.master_list = self.dp.grab_list_range(self.test_counter, self.test_counter+self.hyp.FOOTPRINT+1)
        self.test_counter += 1
        self.batch_counter = 0
        input_data = [k[1:] for k in self.master_list[:-1]]
        return input_data

    def reset_test_counter(self):
        self.test_counter = self.training_set_size

    def next_epoch_valid_waterfall(self):  #this is returning the entire datalist, useful for the "contained" modules that don't use for loops
        if self.validation_set_size ==0:
            raise Exception("You have not initialized the validation set!")
        if self.valid_counter + self.hyp.FOOTPRINT + 1 > self.validation_set_size:
            raise ValueError("you have reached the end of the validation. Please check your code"
                             " for boundary cases. Violation dataset_maker/next_epoch_valid")
        self.master_list = self.dp.grab_list_range(self.valid_counter, self.valid_counter+self.hyp.FOOTPRINT+1)
        self.valid_counter += 1
        self.batch_counter = 0
        input_data = [k[1:] for k in self.master_list[:-1]]
        return input_data

    def clear_valid_counter(self): #this is a public function that clears the validation counter
        self.valid_counter =0

    def clear_counter(self): #resets the counter. This is a private function
        self.counter = 0

    def get_label(self): #returns the label of the current epoch
        return self.master_list[-1][0]

    def next_sample(self): #returns the next sample of the epoch
        if self.batch_counter >=self.hyp.FOOTPRINT:
            raise ValueError("you are infiltrating into key territory! Traceback: dataset_maker/next_sample. "
                             "Violation: batch_counter > self.hyp.FOOTPRINT")
        else:
            carrier = self.master_list[self.batch_counter]
            self.batch_counter += 1
            return carrier