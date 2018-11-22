import tensorflow as tf
from pipeline.dataset_maker import SetMaker
from pipeline.hyperparameters import Hyperparameters
import numpy as np
import csv

hyp = Hyperparameters()
sm = SetMaker()
pbfilename = "2012/v1/models/LSTM_v1_frozen.pb"


with tf.gfile.GFile(pbfilename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def,
                        input_map = None,
                        return_elements = None,
                        name = "")
    input = graph.get_tensor_by_name("placeholders/input_placeholder:0")
    output = graph.get_tensor_by_name("output_gate/BHTO_b_a:0")
    H_last = graph.get_tensor_by_name("placeholders/last_hidden:0")
    current_hidden = graph.get_tensor_by_name("output_gate/next_hidden:0")
    C_last = graph.get_tensor_by_name("placeholders/last_cell:0")
    current_cell = graph.get_tensor_by_name("output_gate/output_presquashing:0")

with tf.Session(graph=graph) as sess:
    sm.create_training_set()
    test = open("2012/v1/GRAPHS/EVALUATE_TEST.csv", "w")
    test_logger = csv.writer(test, lineterminator="\n")
    carrier = ["true_values", "predicted_values", "abs_error"]
    test_logger.writerow(carrier)
    for test in range(hyp.Info.EVAULATE_TEST_SIZE): #this will be replaced later
        print(test)
        sm.next_epoch_test_single_shift()
        label_ = sm.get_label()
        label = np.reshape(label_, [1, 1])
        next_cell = np.zeros(shape=[1, hyp.cell_dim])
        next_hidd = np.zeros(shape=[1, hyp.hidden_dim])

        for counter in range(hyp.FOOTPRINT):
            data = sm.next_sample()
            data = np.reshape(data, [1, 1])
            if counter < hyp.FOOTPRINT-1:
                next_cell, next_hidd = sess.run([current_cell, current_hidden],
                                                feed_dict={input: data, H_last: next_hidd, C_last: next_cell})
            else:
                output_= sess.run(
                    output,
                    feed_dict={input: data, H_last: next_hidd, C_last: next_cell})

                carrier = [label_, output_[0][0], np.abs(label_ - output_)[0][0]]
                test_logger.writerow(carrier)

