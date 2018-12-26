"""Maximilian Du 11-25-18
LSTM implementation with wind data set
Version 2 genetic
this does not use the hyperparameters class anymore, instead using the arguments
THIS DOESN'T SAVE ANYTHING, NOR MAKES ANYTHING.
IT ONLY TELLS YOU THE BEST HYPERPARAMETERS,
YOU TRAIN AGAIN WITH SAVING MECHANISMS IN PLACE LATER
"""
import tensorflow as tf
import numpy as np
from pipeline.dataset_maker import SetMaker
import sys
import csv


hyperparameters = [float(k) for k in sys.argv[1:]] # this extracts all hyperparameters
#format: lr, ftprint, cd, hd
footprint = int(hyperparameters[0])
learning_rate = hyperparameters[1]
cell_dim = int(hyperparameters[2])
hidden_dim = int(hyperparameters[3])
epochs = int(hyperparameters[4])#just a data issue. No data is being destroyed here. I'm just changing it to a compatible type
test_size = int(hyperparameters[5])
SERIAL_NUMBER = int(hyperparameters[6]) #this is for telling which instance this is

sm = SetMaker(footprint)
#this makes the crash file, to b e deleted later
test = open("../Genetic/" + str(SERIAL_NUMBER) + ".csv", "w")
test_logger = csv.writer(test, lineterminator="\n")

#constructing the big weight now
with tf.name_scope("weights_and_biases"):
    W_Forget = tf.Variable(tf.random_normal(shape = [hidden_dim + 1,cell_dim]), name = "forget_weight")
    W_Output = tf.Variable(tf.random_normal(shape=[hidden_dim + 1,cell_dim]), name="output_weight")
    W_Gate = tf.Variable(tf.random_normal(shape=[hidden_dim + 1, cell_dim]), name="gate_weight")
    W_Input = tf.Variable(tf.random_normal(shape=[hidden_dim + 1, cell_dim]), name="input_weight")
    W_Hidden_to_Out = tf.Variable(tf.random_normal(shape=[hidden_dim,1]), name = "outwards_propagating_weight")

    B_Forget = tf.Variable(tf.zeros(shape=[1, cell_dim]), name = "forget_bias")
    B_Output = tf.Variable(tf.zeros(shape=[1, cell_dim]), name="output_bias")
    B_Gate = tf.Variable(tf.zeros(shape=[1, cell_dim]), name="gate_bias")
    B_Input = tf.Variable(tf.zeros(shape=[1,cell_dim]), name="input_bias")
    B_Hidden_to_Out = tf.Variable(tf.zeros(shape=[1,1]), name = "outwards_propagating_bias")

with tf.name_scope("placeholders"):
    Y = tf.placeholder(shape = [1,1], dtype = tf.float32, name = "label") #not used until the last cycle
    init_state = tf.placeholder(shape = [2,1,cell_dim], dtype = tf.float32, name = "initial_states")
    inputs = tf.placeholder(shape = [footprint,1,1], dtype = tf.float32,  name = "input_data")

def step(last_state, X):
    with tf.name_scope("to_gates"):
        C_last, H_last = tf.unstack(last_state)
        concat_input = tf.concat([X, H_last], axis = 1, name = "input_concat") #concatenates the inputs to one vector
        forget_gate = tf.add(tf.matmul(concat_input, W_Forget, name = "f_w_m"),B_Forget, name = "f_b_a") #decides which to drop from cell
        output_gate = tf.add(tf.matmul(concat_input, W_Output, name = "o_w_m"), B_Output, name = "o_b_a") #decides which to reveal to next_hidd/output
        gate_gate = tf.add(tf.matmul(concat_input, W_Gate, name = "g_w_m"), B_Gate, name = "g_b_a") #decides which things to change in cell state
        input_gate = tf.add(tf.matmul(concat_input, W_Input, name = "i_w_m"), B_Input, name = "i_b_a") #decides which of the changes to accept

    with tf.name_scope("non-linearity"): #makes the gates into what they should be
        forget_gate = tf.sigmoid(forget_gate, name = "sigmoid_forget")
        output_gate = tf.sigmoid(output_gate, name="sigmoid_output")
        input_gate = tf.sigmoid(input_gate, name="sigmoid_input")
        gate_gate = tf.tanh(gate_gate, name = "tanh_gate")

    with tf.name_scope("forget_gate"): #forget gate values and propagate
        current_cell = tf.multiply(forget_gate, C_last, name = "forget_gating")

    with tf.name_scope("suggestion_node"): #suggestion gate
        suggestion_box = tf.multiply(input_gate, gate_gate, name = "input_determiner")
        current_cell = tf.add(suggestion_box, current_cell, name = "input_and_gate_gating")

    with tf.name_scope("output_gate"): #output gate values to hidden
        current_cell = tf.tanh(current_cell, name = "output_presquashing")
        current_hidden = tf.multiply(output_gate, current_cell, name="next_hidden")
        states = tf.stack([current_cell, current_hidden])
    return states

with tf.name_scope("forward_roll"):
    states_list = tf.scan(fn = step, elems = inputs, initializer = init_state, name = "scan")
    curr_state = states_list[-1]
    pass_back_state = tf.add([0.0], states_list[0], name = "pass_back_state")

with tf.name_scope("prediction"):
    _, current_hidden = tf.unstack(curr_state)
    raw_output = tf.add(tf.matmul(current_hidden, W_Hidden_to_Out, name="WHTO_w_m"), B_Hidden_to_Out, name="BHTO_b_a")
    output = tf.nn.relu(raw_output, name="output")

with tf.name_scope("loss"):
    loss = tf.square(tf.subtract(output, Y))
    loss = tf.reshape(loss, [], name = "loss")


with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
'''
with tf.name_scope("summaries_and_saver"):
    tf.summary.histogram("W_Forget", W_Forget)
    tf.summary.histogram("W_Input", W_Input)
    tf.summary.histogram("W_Output", W_Output)
    tf.summary.histogram("W_Gate", W_Gate)
    tf.summary.histogram("W_Hidden_to_Out", W_Hidden_to_Out)

    tf.summary.histogram("B_Forget", B_Forget)
    tf.summary.histogram("B_Input", B_Input)
    tf.summary.histogram("B_Output", B_Output)
    tf.summary.histogram("B_Gate", B_Gate)
    tf.summary.histogram("B_Hidden_to_Out", B_Hidden_to_Out)

    tf.summary.scalar("Loss", loss)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sm.create_training_set()
    summary = None #this is just because it was used before
    next_state = np.zeros(shape=[2,1,cell_dim])

    for epoch in range(epochs):
        reset, data = sm.next_epoch_waterfall() #this gets you the entire cow, so to speak
        label = sm.get_label()
        label = np.reshape(label, [1, 1])
        data = np.reshape(data, [footprint,1,1])
        loss_ = 0

        if reset:  # this allows for hidden states to reset after the training set loops back around
            next_state = np.zeros(shape=[2,1,cell_dim])

        next_state, loss_,  _ = sess.run([curr_state, loss, optimizer],
                                                          feed_dict = {inputs:data, Y:label, init_state:next_state})
        '''
        if epoch % 500 == 0:
            print("I finished epoch ", epoch, " out of ", epochs, " epochs")
            print("The absolute value loss for this sample is ", np.sqrt(loss_))
            print("predicted number: ", output_, ", real number: ", label)
            '''
        '''if(epoch % 100 == 0):
            print("This is epoch " + str(epoch) + " and the loss is " + str(loss_))'''
    RMS_loss = 0.0
    next_state = np.zeros(shape=[2, 1,cell_dim])
    print(np.shape(next_state))
    for test in range(test_size):  # this will be replaced later

        data = sm.next_epoch_test_waterfall()
        label_ = sm.get_label()
        label = np.reshape(label_, [1, 1])
        data = np.reshape(data, [footprint, 1, 1])

        next_state, output_, loss_ = sess.run([pass_back_state, output, loss],
                                              # why passback? Because we only shift by one!
                                              feed_dict={inputs: data, Y: label, init_state: next_state})
        RMS_loss += np.sqrt(loss_)
        #carrier = [label_, output_[0][0], np.sqrt(loss_)]
        #test_logger.writerow(carrier)
    RMS_loss = RMS_loss / test_size
    print("test for " + str(SERIAL_NUMBER) + ": rms loss is ", RMS_loss)
    test_logger.writerow([RMS_loss])
print("FINISHED ONE PROGRAM")
exit(0.1)