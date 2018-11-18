import tensorflow as tf
from pipeline import SetMaker
from pipeline import Hyperparameters
import numpy as np
import csv

hyp = Hyperparameters()
sm = SetMaker()
pbfilename = "2012/v2/models_CONTAINED/LSTM_v2_frozen_CONTAINED.pb"


with tf.gfile.GFile(pbfilename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def,
                        input_map = None,
                        return_elements = None,
                        name = "")
    input = graph.get_tensor_by_name("placeholders/input_data:0")
    init_state = graph.get_tensor_by_name("placeholders/initial_states:0")
    output = graph.get_tensor_by_name("prediction/output:0")
    pass_back_state = graph.get_tensor_by_name("forward_roll/pass_back_state:0")

with tf.Session(graph=graph) as sess:
    sm.create_training_set()
    test = open("2012/v2/GRAPHS_CONTAINED/EVALUATE_TEST.csv", "w")
    test_logger = csv.writer(test, lineterminator="\n")
    carrier = ["true_values", "predicted_values", "abs_error"]
    test_logger.writerow(carrier)
    RMS_loss = 0.0
    init_state_ = np.zeros(shape=[2, 1, hyp.cell_dim])
    for i in range(hyp.Info.TEST_SIZE):  # this will be replaced later
        data = sm.next_epoch_test_waterfall()
        label_ = sm.get_label()
        label = np.reshape(label_, [1, 1])
        print(i)
        data = np.reshape(data, [hyp.FOOTPRINT, 1, 1])


        init_state_ , output_= sess.run([pass_back_state, output],
                                                feed_dict = {input: data, init_state: init_state_})

        loss_ = np.square(output_[0][0] - label_)
        RMS_loss += np.sqrt(loss_)
        carrier = [label_, output_[0][0], np.sqrt(loss_)]
        test_logger.writerow(carrier)
    RMS_loss = RMS_loss / hyp.Info.TEST_SIZE
    print("test: rms loss is ", RMS_loss)
    test_logger.writerow(["final absolute loss average", RMS_loss])

