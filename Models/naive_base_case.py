from pipeline.dataset_maker import SetMaker
from pipeline.hyperparameters import Hyperparameters
import csv

hyp = Hyperparameters()
sm = SetMaker(9)

sm.create_training_set()

sm.set_test_number(81072)
test = open("../Graphs_and_Results/Naive_case_.csv", "w")
test_logger = csv.writer(test, lineterminator="\n")
test_logger.writerow(["true_values", "predicted_values", "abs_loss"])
for i in range(hyp.Info.EVAULATE_TEST_SIZE):
    m = list()
    sm.next_epoch_test_single_shift()
    for k in range(9):

        m.append(sm.next_sample())
    truth = sm.get_label()
    naive_result = m[8]
    loss = abs(naive_result-truth)
    test_logger.writerow([truth, naive_result, loss])


