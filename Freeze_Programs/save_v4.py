from tensorflow.python.tools import freeze_graph
version = 2
version_to_keep = 159850

MODEL_NAME = 'LSTM_v' + str(version) + '_genetic_frozen_FORE'
#MODEL_NAME = 'gru_c_genetic_frozen'
# Freeze the graph

input_graph_path = '../Graphs_and_Results/lstm_v' + str(version) + '_c_class_FORE/GRAPHS/graph.pbtxt'
#input_graph_path = '../Graphs_and_Results/gru_c_class/GRAPHS/graph.pbtxt'


checkpoint_path = '../Graphs_and_Results/lstm_v' + str(version) + '_c_class_FORE/models/v' + str(version) + 'Genetic_FORE-' + str(version_to_keep)
#checkpoint_path = '../Graphs_and_Results/gru_c_class/models/GRUGenetic-' + str(version_to_keep)
input_saver_def_path = ''
input_binary = False
output_node_names = 'prediction/output, forward_roll/pass_back_state'
restore_op_name = 'save/restore_all'
filename_tensor_name = 'save/Const:0'
output_frozen_graph_name = '../Graphs_and_Results/lstm_v' + str(version) + '_c_class_FORE/'+MODEL_NAME+'.pb'
#output_frozen_graph_name = '../Graphs_and_Results/gru_c_class/'+MODEL_NAME+'.pb'

clear_devices = True


freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, '')

print("I'm done.")