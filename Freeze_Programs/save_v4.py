import tensorflow as tf
from tensorflow.python.tools import freeze_graph
MODEL_NAME = 'LSTM_v2_genetic_frozen'

# Freeze the graph

input_graph_path = "../Graphs_and_Results/lstm_v2_c_CSV_FED/GRAPHS/graph.pbtxt"
checkpoint_path = "../Graphs_and_Results/lstm_v2_c_CSV_FED/models/V2Genetic-80000"
input_saver_def_path = ""
input_binary = False
output_node_names = "prediction/output, forward_roll/pass_back_state"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = '../Graphs_and_Results/lstm_v2_c_CSV_FED/'+MODEL_NAME+'.pb'

clear_devices = True


freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")

print("I'm done.")