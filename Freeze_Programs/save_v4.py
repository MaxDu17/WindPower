import tensorflow as tf
from tensorflow.python.tools import freeze_graph
MODEL_NAME = 'LSTM_v2_frozen_CONTAINED'

# Freeze the graph

input_graph_path = "2012/v2/GRAPHS_CONTAINED/graph.pbtxt"
checkpoint_path = "2012/v2/models_CONTAINED/LSTMv2-60000"
input_saver_def_path = ""
input_binary = False
output_node_names = "prediction/output, forward_roll/pass_back_state"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = '2012/v2/models_CONTAINED/'+MODEL_NAME+'.pb'

clear_devices = True


freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")