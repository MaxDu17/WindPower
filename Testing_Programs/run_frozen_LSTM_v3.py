import tensorflow as tf
from pipeline.dataset_maker import SetMaker
from pipeline.hyperparameters import Hyperparameters
import numpy as np
import csv

hyp = Hyperparameters()
version = 6
MODEL_NAME = 'LSTM_v' + str(version) + '_genetic_frozen'
CSV_NAME = 'lstm_v' + str(version) + '_c_classbest'
k = open("../Genetic/" + CSV_NAME + ".csv", "r")

hyp_list =  list(csv.reader(k)) #extracing the first data point from the csv file
footprint = int(hyp_list[0][0])
hidden_dim =  int(hyp_list[0][2])
sm = SetMaker(footprint)
labels = list()
outputs = list()

pbfilename = '../Graphs_and_Results/lstm_v' + str(version) + '_c_class/'+MODEL_NAME+'.pb'


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
    test = open('../Graphs_and_Results/lstm_v' + str(version) + '_c_class/GRAPHS/EVALUATE_TEST.csv', "w")
    test_logger = csv.writer(test, lineterminator="\n")
    carrier = ["true_values", "predicted_values", "abs_error"]
    test_logger.writerow(carrier)
    RMS_loss = 0.0
    init_state_ = np.zeros(shape=[2, 1, hidden_dim])
    for i in range(hyp.Info.TEST_SIZE):  # this will be replaced later
        data = sm.next_epoch_test_waterfall()
        label_ = sm.get_label()
        label = np.reshape(label_, [1, 1])
        print(i)
        data = np.reshape(data, [footprint, 1, 1])


        init_state_ , output_= sess.run([pass_back_state, output],
                                                feed_dict = {input: data, init_state: init_state_})

        loss_ = np.square(output_[0][0] - label_)
        labels.append(label_)
        outputs.append(output_[0][0])
        RMS_loss += np.sqrt(loss_)
        carrier = [label_, output_[0][0], np.sqrt(loss_)]
        test_logger.writerow(carrier)

    RMS_loss = RMS_loss / hyp.Info.TEST_SIZE
    print("test: rms loss is ", RMS_loss)

######finding naive coeficient###########
big_total_normal = 0
for i in range(len(outputs)):
    big_total_normal += (np.abs(outputs[i] - labels[i]))

outputs = list(np.roll(outputs, -1))
del outputs[-1]
del labels[-1]

big_total_shift = 0
for i in range(len(outputs)):
    big_total_shift += (np.abs(outputs[i] - labels[i]))

print(big_total_shift)
print(big_total_normal)

naive_coeficient = big_total_normal - big_total_shift
print("Naive coeficient: " + str(naive_coeficient))
file = open('../Graphs_and_Results/lstm_v' + str(version) + '_c_class/GRAPHS/naivecoeff.txt', 'w')
file.write(str(naive_coeficient))
