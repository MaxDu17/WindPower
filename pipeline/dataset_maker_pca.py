from pipeline.data_feeder_pca import DataParser_Forecast
from pipeline.hyperparameters import Hyperparameters

class SetMaker_Forecast:
    def __init__(self, FOOTPRINT): #initializing variables used in calculation
        self.dp = DataParser_Forecast()
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
        self.FOOTPRINT = FOOTPRINT #this will allow genetic feeding
        self.custom_test = 81072  # for recording purposes

    def create_training_set(self): #initializing statement for training
        self.training_set_size = int(self.hyp.TRAIN_PERCENT * self.dp.dataset_size())
        self.test_counter = self.training_set_size

    def create_validation_set(self): #initializing phrase for validation
        self.validation_set_size = int(self.hyp.VALIDATION_PERCENT * self.dp.dataset_size()) #just casting to whole #

    def set_test_number(self, number):
        self.test_counter = number


    def get_test_number(self):
        return self.test_counter-1000

    def next_epoch(self): #this gets a new training epoch, retrns status on resetting the states
        carrier = False
        self.master_list = list()
        if self.counter + self.FOOTPRINT+1 > self.training_set_size:
            self.clear_counter()
            carrier = True
            print("wraparound")
        self.master_list = self.dp.grab_list_range(self.counter, self.counter+self.FOOTPRINT+1)
        self.counter += self.FOOTPRINT
        self.batch_counter = 0
        return carrier

    def next_epoch_waterfall(self): #this just returns the entire sequence. DO NOT CALL NEXT_SAMPLE WITH THIS.
        carrier = False
        self.master_list = list()
        if self.counter + self.FOOTPRINT+1 > self.training_set_size:
            self.clear_counter()
            carrier = True #this effectively tells the network to reset
            print("wraparound")

        self.master_list = self.dp.grab_list_range(self.counter, self.counter+self.FOOTPRINT+1)
        self.pca_only = list()
        for k in self.master_list:
            self.pca_only.append(k[0])
        self.counter += self.FOOTPRINT
        self.batch_counter = 0
        return carrier, self.pca_only[:-1]

    def next_epoch_test(self): #this jumps a footprint to test. Biased estimator, and so is depreciated.
        if self.test_counter == 0:
            raise Exception("you forgot to initialize the test_counter! Execute create_training_set")
        if self.test_counter + self.FOOTPRINT + 1 > self.dp.dataset_size():
            raise ValueError("you have reached the end of the test set. Violation dataset_maker/next_epoch_test")
        self.master_list = list()
        self.master_list = self.dp.grab_list_range(self.test_counter, self.test_counter + self.FOOTPRINT + 1)
        self.test_counter += self.FOOTPRINT
        self.batch_counter = 0

    def next_epoch_test_waterfall(self): #this is nextepochtestsingleshift but with a waterfall

        if self.test_counter == 0:
            raise Exception("you forgot to initialize the test_counter! Execute create_training_set")
        if self.test_counter + self.FOOTPRINT + 1 > self.dp.dataset_size():
            raise ValueError("you have reached the end of the test set. Violation dataset_maker/next_epoch_test")
        self.master_list = list()
        self.master_list = self.dp.grab_list_range(self.test_counter, self.test_counter + self.FOOTPRINT + 1)
        self.pca_only = list()
        for k in self.master_list:
            self.pca_only.append(k[0])
        self.counter += self.FOOTPRINT
        self.batch_counter = 0
        self.test_counter += 1
        return self.pca_only[:-1]


    def next_epoch_valid(self): #next validation epoch. Note that this is imperative; it doesn't return anything
        if self.valid_counter + self.FOOTPRINT + 1 > self.validation_set_size:
            raise ValueError("you have reached the end of the validation. Please check your code"
                             " for boundary cases. Violation dataset_maker/next_epoch_valid")
        self.master_list = list()
        self.master_list = self.dp.grab_list_range(self.valid_counter, self.valid_counter + self.FOOTPRINT + 1)
        self.valid_counter += 1
        self.batch_counter = 0

    def next_epoch_valid_waterfall(self):  #this is returning the entire datalist, useful for the "contained" modules that don't use for loops
        if self.valid_counter + self.FOOTPRINT + 1 > self.validation_set_size:
            raise ValueError("you have reached the end of the validation. Please check your code"
                             " for boundary cases. Violation dataset_maker/next_epoch_valid")
        self.master_list = list()
        self.master_list = self.dp.grab_list_range(self.valid_counter, self.valid_counter + self.FOOTPRINT + 1)
        self.pca_only = list()
        for k in self.master_list:
            self.pca_only.append(k[0])
        self.counter += self.FOOTPRINT
        self.batch_counter = 0
        self.valid_counter +=1
        return self.pca_only[:-1]

    def clear_valid_counter(self): #this is a public function that clears the validation counter
        self.valid_counter =0

    def reset_test_counter(self):
        self.test_counter = self.training_set_size

    def clear_counter(self): #resets the counter. This is a private function
        self.counter = 0

    def get_label(self): #returns the label of the current epoch
        return self.master_list[-1][1]

    def next_sample(self): #returns the next sample of the epoch
        if self.batch_counter >=self.FOOTPRINT:
            raise ValueError("you are infiltrating into key territory! Traceback: dataset_maker/next_sample. "
                             "Violation: batch_counter > self.FOOTPRINT")
        else:
            carrier = self.master_list[self.batch_counter]
            self.batch_counter += 1
            return carrier

    def next_sample_list(self): #this is the basic version of the waterfall
        return self.master_list

    def return_split_lists(self):
        self.from_run = self.dp.grab_list_range(self.hyp.RUN_PROMPT, self.hyp.Info.RUN_TEST_SIZE)
        self.from_start = self.dp.grab_list_range(0, self.hyp.RUN_PROMPT)
        return self.from_run, self.from_start