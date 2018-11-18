class Hyperparameters: #the class that defines the hyperparameters is here
    FOOTPRINT = 9 #how many steps back you take. This is a critical adjustment point
    LEARNING_RATE = 0.001
    EPOCHS = 80001
    TRAIN_PERCENT = 0.6
    VALIDATION_PERCENT = 0.002 #nullifies for now
    VALIDATION_NUMBER = 60
    cell_dim = 75
    hidden_dim = 75
    TEST = True
    SAVER_JUMP = 2000
    SUMMARY_JUMP = 50
    RUN_PROMPT = 25
    STD = 0.5
    SEED = 76137
    MEAN = 0
    FINALJUMP = 10 #this is how many tests it will spit out during the last portion of training

    class Info: #not used in real code, just as metadata
        DATASET_SIZE = 150120
        TRAIN_SIZE = 63072
        #TEST_SIZE = 42048
        TEST_SIZE = 1000
        TEST_START = 63072
        EVAULATE_TEST_SIZE = 1000
        RUN_TEST_SIZE = 100
        VALID_SIZE = 210
        VALID_SIZE_LIMIT = 200 #highest number you can ask for
        #TEST_SIZE_LIMIT = 42038 #this is the last number you can ask for
        LOSS_FUNCTION = "squared_error"

