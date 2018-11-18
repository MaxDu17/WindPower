import tensorflow as tf
from pipeline import SetMaker
from pipeline import Hyperparameters
import numpy as np
import csv

hyp = Hyperparameters()
sm = SetMaker()
pbfilename = "2012/v7/models/LSTM_v7_frozen.pb"

with tf.gfile.GFile(pbfilename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def,
                        input_map = None,
                        return_elements = None,
                        name = "")
    input = graph.get_tensor_by_name("placeholders/input_placeholder:0")
    output = graph.get_tensor_by_name("output_gate/output:0")
    H_last = graph.get_tensor_by_name("placeholders/last_hidden:0")
    current_hidden = graph.get_tensor_by_name("output_gate/next_hidden:0")
    C_last = graph.get_tensor_by_name("placeholders/last_cell:0")
    current_cell = graph.get_tensor_by_name("output_gate/cell_squashing:0")


path_name_root = "C:/Users/Max Du/Dropbox/My Academics/CSIRE/data 2012/"

for i in range(12):
    sm = SetMaker()
    path_name = path_name_root + str(i+1) + ".csv"
    sm.use_foreign(path_name)
    csv_name = "2012/v7/FOREIGN_LOG/FOREIGN_TEST_" + str(i+1) + ".csv"
    with tf.Session(graph=graph) as sess:
        sm.create_training_set()
        test = open(csv_name, "w")
        test_logger = csv.writer(test, lineterminator="\n")
        carrier = ["true_values", "predicted_values", "abs_error"]
        test_logger.writerow(carrier)
        RMS_loss = 0.0
        next_cell = np.zeros(shape=[1, hyp.cell_dim])
        next_hidd = np.zeros(shape=[1, hyp.hidden_dim])
        hidden_saver = list()
        cell_saver = list()
        for test in range(hyp.Info.EVAULATE_TEST_SIZE):  # this will be replaced later
            print(test)
            sm.next_epoch_test_single_shift()
            label_ = sm.get_label()
            # this gets each 10th
            for counter in range(hyp.FOOTPRINT):
                data = sm.next_sample()
                data = np.reshape(data, [1, 1])
                if counter < hyp.FOOTPRINT - 1:

                    next_cell, next_hidd = sess.run([current_cell, current_hidden],
                                                    feed_dict={input: data, H_last: next_hidd, C_last: next_cell})
                    if counter == 0:
                        hidden_saver = next_hidd  # saves THIS state for the next round
                        cell_saver = next_cell
                else:
                    next_cell, next_hidd, output_ = sess.run(
                        [current_cell, current_hidden, output],
                        feed_dict={input: data, H_last: next_hidd, C_last: next_cell})

                    carrier = [label_, output_[0][0], np.sqrt(np.square((label_ - output_)[0][0]))]
                    test_logger.writerow(carrier)
            next_cell = cell_saver
            next_hidden = hidden_saver







