import tensorflow as tf
from pipeline import SetMaker
from pipeline import Hyperparameters
import numpy as np
import csv

hyp = Hyperparameters()
sm = SetMaker()
pbfilename = "2012/v8/models/LSTM_v8_frozen.pb"


with tf.gfile.GFile(pbfilename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def,
                        input_map = None,
                        return_elements = None,
                        name = "")
    #Y = graph.get_tensor_by_name("placeholders/label:0")
    inputs = graph.get_tensor_by_name("placeholders/input_data:0")
    init_state_1 = graph.get_tensor_by_name("placeholders/initial_states_1:0")
    init_state_2 = graph.get_tensor_by_name("placeholders/initial_states_2:0")
    output = graph.get_tensor_by_name("layer_2_propagation/output:0")
    pass_back_state_2 = graph.get_tensor_by_name("layer_2_propagation/pass_back_state_2:0")
    pass_back_state_1 = graph.get_tensor_by_name("layer_1_propagation/pass_back_state_1:0")
    #loss = graph.get_tensor_by_name("loss/loss:0")

with tf.Session(graph=graph) as sess:
    init_state_1_ = init_state_2_ = np.zeros(shape=[2, 1, hyp.cell_dim])
    sm.create_training_set()
    test = open("2012/v8/GRAPHS/EVALUATE_TEST.csv", "w")
    test_logger = csv.writer(test, lineterminator="\n")
    carrier = ["true_values", "predicted_values", "abs_error"]
    test_logger.writerow(carrier)
    RMS_loss = 0.0
    for test in range(hyp.Info.TEST_SIZE):  # this will be replaced later

        data = sm.next_epoch_test_waterfall()
        label_ = sm.get_label()
        label = np.reshape(label_, [1, 1])
        data = np.reshape(data, [hyp.FOOTPRINT, 1, 1])

        init_state_1_, init_state_2_, output_ = sess.run([pass_back_state_1, pass_back_state_2, output], # why passback? Because we only shift by one!
                                                       feed_dict={init_state_1: init_state_1_,
                                                                  init_state_2: init_state_2_, inputs: data})
        RMS_loss += np.abs(label_-output_[0][0])
        carrier = [label_, output_[0][0], np.abs(label_-output_[0][0])]
        test_logger.writerow(carrier)
    RMS_loss = RMS_loss / hyp.Info.TEST_SIZE
    print("test: rms loss is ", RMS_loss)
    test_logger.writerow(["final adaptive loss average", RMS_loss])
