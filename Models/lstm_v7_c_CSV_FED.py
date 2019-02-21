"""Maximilian Du 1-5-19
this takes in a csv and trains the FIRST data point on it
"""
import tensorflow as tf
import numpy as np
from pipeline.dataset_maker import SetMaker
from pipeline.hyperparameters import Hyperparameters
import os
import sys
import csv

NAME = "lstm_v7_c_class" #this is the name of the python file for logging purposes

k = open("../Genetic/" + str(NAME) + "best.csv", "r")

hyp_list =  list(csv.reader(k)) #extracing the first data point from the csv file

FOOTPRINT = int(hyp_list[0][0])
LEARNING_RATE = float(hyp_list[0][1])
hidden_dim = cell_dim = int(hyp_list[0][2])
sm = SetMaker(FOOTPRINT)
hyp = Hyperparameters() # this is used later for non-changing hyperparameters
epochs = hyp.EPOCHS_LARGE #this is the epochs setting

if len(sys.argv) > 1:
    epochs = int(sys.argv[1]) #this allows us to provide an arbitrary training size


#constructing the big weight now
with tf.name_scope("weights_and_biases"):
    W_Forget_and_Input = tf.Variable(tf.random_normal(shape=[hidden_dim + cell_dim + 1, cell_dim]),
                                     name="forget_and_input_weight")  # note that forget_and_input actually works for forget, and the input is the inverse
    W_Output = tf.Variable(tf.random_normal(shape=[hidden_dim + cell_dim + 1, cell_dim]), name="output_weight")
    W_Gate = tf.Variable(tf.random_normal(shape=[hidden_dim + cell_dim + 1, cell_dim]), name="gate_weight")

    W_Hidden_to_Out = tf.Variable(tf.random_normal(shape=[hidden_dim, 1]), name="outwards_propagating_weight")

    B_Forget_and_Input = tf.Variable(tf.zeros(shape=[1, cell_dim]), name="forget_and_input_bias")
    B_Output = tf.Variable(tf.zeros(shape=[1, cell_dim]), name="output_bias")
    B_Gate = tf.Variable(tf.zeros(shape=[1, cell_dim]), name="gate_bias")
    B_Hidden_to_Out = tf.Variable(tf.zeros(shape=[1, 1]), name="outwards_propagating_bias")

with tf.name_scope("placeholders"):
    Y = tf.placeholder(shape=[1, 1], dtype=tf.float32, name="label")  # not used until the last cycle
    init_state = tf.placeholder(shape=[2, 1, cell_dim], dtype=tf.float32, name="initial_states")
    inputs = tf.placeholder(shape=[FOOTPRINT, 1, 1], dtype=tf.float32, name="input_data")


def step(last_state, X):
    C_last, H_last = tf.unstack(last_state)
    with tf.name_scope("to_gates"):
        concat_input = tf.concat([X, H_last, C_last], axis=1,
                                 name="input_concat")  # concatenates the inputs to one vector
        forget_gate = tf.add(tf.matmul(concat_input, W_Forget_and_Input, name="f_w_m"), B_Forget_and_Input,
                             name="f_b_a")  # decides which to drop from cell

        gate_gate = tf.add(tf.matmul(concat_input, W_Gate, name="g_w_m"), B_Gate,
                           name="g_b_a")  # decides which things to change in cell state

    with tf.name_scope("non-linearity"):  # makes the gates into what they should be
        forget_gate = tf.sigmoid(forget_gate, name="sigmoid_forget")

        forget_gate_negated = tf.scalar_mul(-1, forget_gate)  # this has to be here because it is after the nonlin
        input_gate = tf.add(tf.ones([1, cell_dim]), forget_gate_negated, name="making_input_gate")
        input_gate = tf.sigmoid(input_gate, name="sigmoid_input")

        gate_gate = tf.tanh(gate_gate, name="tanh_gate")

    with tf.name_scope("forget_gate"):  # forget gate values and propagate

        current_cell = tf.multiply(forget_gate, C_last, name="forget_gating")

    with tf.name_scope("suggestion_node"):  # suggestion gate
        suggestion_box = tf.multiply(input_gate, gate_gate, name="input_determiner")
        current_cell = tf.add(suggestion_box, current_cell, name="input_and_gate_gating")

    with tf.name_scope("output_gate"):  # output gate values to hidden

        concat_output_input = tf.concat([X, H_last, current_cell], axis=1,
                                        name="input_concat")  # concatenates the inputs to one vector #here, the processed current cell is concatenated and prepared for output
        output_gate = tf.add(tf.matmul(concat_output_input, W_Output, name="o_w_m"), B_Output,
                             name="o_b_a")  # we are making the output gates now, with the peephole.
        output_gate = tf.sigmoid(output_gate,
                                 name="sigmoid_output")  # the gate is complete. Note that the two lines were supposed to be back in "to gates" and "non-linearity", but it is necessary to put it here
        current_cell = tf.tanh(current_cell,
                               name="cell_squashing")  # squashing the current cell, branching off now. Note the underscore, means saving a copy.
        current_hidden = tf.multiply(output_gate, current_cell,
                                     name="next_hidden")  # we are making the hidden by element-wise multiply of the squashed states

        states = tf.stack([current_cell, current_hidden])  # this states is just about ready for outward propagation

    return states


with tf.name_scope("forward_roll"):
    states_list = tf.scan(fn=step, elems=inputs, initializer=init_state,
                          name="scan")  # funs step until inputs are gone
    curr_state = states_list[-1]  # used for the next time step
    pass_back_state = tf.add([0.0], states_list[0],
                             name="pass_back_state")  # this is just making it a compute for tensorboard vis

with tf.name_scope("prediction"):
    _, current_hidden = tf.unstack(curr_state)  # this gets the current hidden layer
    raw_output = tf.add(tf.matmul(current_hidden, W_Hidden_to_Out, name="WHTO_w_m"), B_Hidden_to_Out,
                        name="BHTO_b_a")  # gets raw output
    output = tf.nn.relu(raw_output, name="output")

with tf.name_scope("loss"):
    loss = tf.square(tf.subtract(output, Y))
    loss = tf.reduce_sum(loss)

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

with tf.name_scope("summaries_and_saver"): #these are for tensorboard display
    tf.summary.histogram("W_Forget_and_Input", W_Forget_and_Input)
    tf.summary.histogram("W_Output", W_Output)
    tf.summary.histogram("W_Gate", W_Gate)
    tf.summary.histogram("W_Hidden_to_Out", W_Hidden_to_Out)

    tf.summary.histogram("B_Forget_and_Input", B_Forget_and_Input)
    tf.summary.histogram("B_Output", B_Output)
    tf.summary.histogram("B_Gate", B_Gate)
    tf.summary.histogram("B_Hidden_to_Out", B_Hidden_to_Out)

    tf.summary.scalar("Loss", loss)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=9)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #this initializes the compute nodes
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('../Graphs_and_Results/' + NAME+ '/models/'))
    if ckpt and ckpt.model_checkpoint_path:
        query = input("checkpoint detected! Would you like to restore from <" + ckpt.model_checkpoint_path + "> ?(y or n)\n")
        if query == 'y':
            saver.restore(sess, ckpt.model_checkpoint_path)
            if np.sum(B_Forget_and_Input.eval()) != 0: #this checks for restored session
                print("session restored!")
        else:
            print("session discarded!")

    log_loss = open("../Graphs_and_Results/" + NAME + "/GRAPHS/LOSS.csv", "w") #this is the loss over time
    validation = open("../Graphs_and_Results/" + NAME + "/GRAPHS/VALIDATION.csv", "w") #this is the validation
    test = open("../Graphs_and_Results/" + NAME + "/GRAPHS/TEST.csv", "w") #this is the test file
    logger = csv.writer(log_loss, lineterminator="\n")
    validation_logger = csv.writer(validation, lineterminator="\n")
    test_logger = csv.writer(test, lineterminator="\n")

    sm.create_training_set() #we initialize the training set


    tf.train.write_graph(sess.graph_def, '../Graphs_and_Results/' + NAME + '/GRAPHS/', 'graph.pbtxt') #this makes the pbtxt needed for freeze
    writer = tf.summary.FileWriter("../Graphs_and_Results/" + NAME + "/GRAPHS/", sess.graph) #this will write summary tensorboard

    summary = None
    next_state = np.zeros(shape=[2,1,cell_dim]) #this initializes the initial "next state"

    for epoch in range(epochs):
        reset, data = sm.next_epoch_waterfall() #this gets you the entire data chunk
        label = sm.get_label() #this is the answer key
        label = np.reshape(label, [1, 1]) #reshaping for data transfer
        data = np.reshape(data, [FOOTPRINT,1,1])
        loss_ = 0

        if reset:  # this allows for hidden states to reset after the training set loops back around
            next_state = np.zeros(shape=[2,1,cell_dim])

        ################# this is the running command ####################################################
        next_state, output_, loss_, summary, _ = sess.run([curr_state, output, loss, summary_op, optimizer],
                                                          feed_dict = {inputs:data, Y:label, init_state:next_state})
        #########################################################################################################

        logger.writerow([loss_])

        if epoch % 200 == 0: #display current error
            writer.add_summary(summary, global_step=epoch)
            print("I finished epoch ", epoch, " out of ", epochs, " epochs")
            print("The absolute value loss for this sample is ", np.sqrt(loss_))
            print("predicted number: ", output_, ", real number: ", label)

        if epoch % 50 == 0 and epoch > epochs-(50*hyp.FINALJUMP): #this finds entropic local minima at the end and saves them
            test_local_ = open("../Graphs_and_Results/" + NAME + "/models/" + str(epoch) + ".csv", 'w')
            test_local = csv.writer(test_local_, lineterminator='\n')

            saver.save(sess, "../Graphs_and_Results/" + NAME + "/models/v7Genetic", global_step=epoch)

            RMS_loss = 0.0
            next_state_test = np.zeros(shape=[2, 1, cell_dim]) #initializations
            carrier = ["true_values", "predicted_values", "abs_error"]
            test_local.writerow(carrier)
            sm.reset_test_counter()

            for test in range(hyp.Info.TEST_SIZE):  # this will be replaced later
                data = sm.next_epoch_test_waterfall()
                label_ = sm.get_label()
                label = np.reshape(label_, [1, 1])
                #data = np.reshape(data, [FOOTPRINT, 1, 6])
                data = np.reshape(data, [FOOTPRINT, 1, 1])
                next_state_test, output_, loss_ = sess.run([pass_back_state, output, loss],
                                                           # why passback? Because we only shift by one!
                                                           feed_dict={inputs: data, Y: label, init_state: next_state_test})
                RMS_loss += np.sqrt(loss_)
                carrier = [label_, output_[0][0], np.sqrt(loss_)]
                test_local.writerow(carrier)
            RMS_loss = RMS_loss / hyp.Info.TEST_SIZE
            print("doing some rapid tests: this one had loss of " + str(RMS_loss))
            test_local_.close()

####################################VALIDATION#######################################
        if epoch % 2000 == 0 and epoch > 498: #this is the validation step
            saver.save(sess, "../Graphs_and_Results/" + NAME + "/models/v7Genetic", global_step=epoch)
            print("---------------------saved model-------------------------")

            next_state_hold = next_state #this "pauses" the training that is happening right now.
            sm.create_validation_set()
            RMS_loss = 0.0
            next_state = np.zeros(shape=[2, 1, cell_dim])
            for i in range(hyp.VALIDATION_NUMBER):
                data = sm.next_epoch_valid_waterfall()
                label_ = sm.get_label()
                label = np.reshape(label_, [1, 1])
                data = np.reshape(data, [FOOTPRINT, 1, 1])

                next_state, loss_ = sess.run([pass_back_state, loss], #why passback? Because we only shift by one!
                                               feed_dict = {inputs:data, Y:label, init_state:next_state})
                RMS_loss += np.sqrt(loss_)
            sm.clear_valid_counter()

            RMS_loss = RMS_loss / hyp.VALIDATION_NUMBER
            print("validation: RMS loss is ", RMS_loss)
            validation_logger.writerow([epoch, RMS_loss])

            next_state = next_state_hold #restoring past point...

#################################TESTING###############################################
    RMS_loss = 0.0
    next_state = np.zeros(shape=[2, 1, cell_dim])
    print(np.shape(next_state))
    sm.reset_test_counter()
    for test in range(hyp.Info.TEST_SIZE):  # this will be replaced later

        data = sm.next_epoch_test_waterfall()
        label_ = sm.get_label()
        label = np.reshape(label_, [1, 1])
        data = np.reshape(data, [FOOTPRINT, 1, 1])

        next_state, output_, loss_ = sess.run([pass_back_state, output, loss],  # why passback? Because we only shift by one!
                                     feed_dict={inputs: data, Y: label, init_state: next_state})
        RMS_loss += np.sqrt(loss_)
        carrier = [label_, output_[0][0], np.sqrt(loss_)]
        test_logger.writerow(carrier)
    RMS_loss = RMS_loss / hyp.Info.TEST_SIZE
    print("test: rms loss is ", RMS_loss)
    test_logger.writerow(["Loss average", RMS_loss])