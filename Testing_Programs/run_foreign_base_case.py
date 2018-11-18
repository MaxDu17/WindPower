import tensorflow as tf
from pipeline import SetMaker
from pipeline import Hyperparameters
import numpy as np
import csv

hyp = Hyperparameters()
sm = SetMaker()
path_name_root = "C:/Users/Max Du/Dropbox/My Academics/CSIRE/data 2012/"

for i in range(12):
    sm = SetMaker()
    path_name = path_name_root + str(i+1) + ".csv"
    sm.use_foreign(path_name)
    csv_name = "2012/v0/FOREIGN_LOG/FOREIGN_TEST_" + str(i+1) + ".csv"
    sm.create_training_set()
    test = open(csv_name, "w")
    test_logger = csv.writer(test, lineterminator="\n")
    carrier = ["true_values", "predicted_values", "abs_error"]
    test_logger.writerow(carrier)
    print(i)
    for i in range(hyp.Info.EVAULATE_TEST_SIZE):
        m = list()
        sm.next_epoch_test_single_shift()
        for k in range(9):
            m.append(sm.next_sample())
        truth = sm.get_label()
        naive_result = m[8]
        loss = abs(naive_result - truth)
        test_logger.writerow([truth, naive_result, loss])










