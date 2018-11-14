"""Maximilian Du 7-17-18
LSTM implementation with wind data set
Version 9 changes:
added weather data
also added picking
"""
import tensorflow as tf
import numpy as np
from pipeline import SetMaker_Weather
from pipeline import Hyperparameters
import os
import csv

sm = SetMaker_Weather()
hyp = Hyperparameters()
#constructing the big weight now

with tf.name_scope("weights_and_biases"):
    W_Forget = tf.Variable(tf.random_normal(shape = [hyp.hidden_dim + 6,hyp.cell_dim]), name = "forget_weight")
    W_Output = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 6,hyp.cell_dim]), name="output_weight")
    W_Gate = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 6, hyp.cell_dim]), name="gate_weight")
    W_Input = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 6, hyp.cell_dim]), name="input_weight")
    W_Hidden_to_Out = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim,1]), name = "outwards_propagating_weight")


    B_Forget = tf.Variable(tf.scalar_mul(5,tf.ones(shape=[1, hyp.cell_dim])), name = "forget_bias")
    B_Output = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="output_bias")
    B_Gate = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="gate_bias")
    B_Input = tf.Variable(tf.zeros(shape=[1,hyp.cell_dim]), name="input_bias")
    B_Hidden_to_Out = tf.Variable(tf.zeros(shape=[1, 1]), name="outwards_propagating_bias")

with tf.name_scope("placeholders"):
    Y = tf.placeholder(shape = [1,1], dtype = tf.float32, name = "label") #not used until the last cycle
    init_state = tf.placeholder(shape = [2,1,hyp.cell_dim], dtype = tf.float32, name = "initial_states")
    inputs = tf.placeholder(shape = [hyp.FOOTPRINT,1,6], dtype = tf.float32,  name = "input_data")

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
        current_cell = tf.tanh(current_cell, name = "to_hidden")
        current_hidden = tf.multiply(output_gate, current_cell, name="next_hidden")
        states = tf.stack([current_cell, current_hidden])
    return states

with tf.name_scope("forward_roll"):
    states_list = tf.scan(fn = step, elems = inputs, initializer = init_state, name = "scan")
    curr_state = states_list[-1]
    pass_back_state = tf.add([0.0], states_list[0], name = "pass_back_state")

with tf.name_scope("prediction"):
    _cell, current_hidden = tf.unstack(curr_state)
    raw_output = tf.add(tf.matmul(current_hidden, W_Hidden_to_Out, name="WHTO_w_m"), B_Hidden_to_Out, name="BHTO_b_a")
    output = tf.nn.relu(raw_output, name="output")

with tf.name_scope("loss"):
    loss = tf.abs(tf.subtract(output, Y))
    loss = tf.reshape(loss, [], name = "loss")

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=hyp.LEARNING_RATE).minimize(loss)

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
    tf.summary.histogram("Current_cell", _cell)
    tf.summary.scalar("Loss", loss)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('evaluation/continuous/currweather/RELU'))
    if ckpt and ckpt.model_checkpoint_path:
        query = input("checkpoint detected! Would you like to restore from <" + ckpt.model_checkpoint_path + "> ?(y or n)\n")
        if query == 'y':
            saver.restore(sess, ckpt.model_checkpoint_path)
            if np.sum(B_Forget.eval()) != 0:
                print("session restored!")
        else:
            print("session discarded!")

    log_loss = open("evaluation/continuous/currweather/RELU/GRAPHS/LOSS.csv", "w")
    validation = open("evaluation/continuous/currweather/RELU/GRAPHS/VALIDATION.csv", "w")
    test = open("evaluation/continuous/currweather/RELU/GRAPHS/TEST.csv", "w")
    logger = csv.writer(log_loss, lineterminator="\n")
    validation_logger = csv.writer(validation, lineterminator="\n")
    test_logger = csv.writer(test, lineterminator="\n")

    sm.create_training_set()


    tf.train.write_graph(sess.graph_def, 'evaluation/continuous/currweather/RELU/GRAPHS/', 'graph.pbtxt')
    writer = tf.summary.FileWriter("evaluation/continuous/currweather/RELU/GRAPHS/", sess.graph)

    summary = None
    next_state = np.zeros(shape=[2,1,hyp.cell_dim])

    for epoch in range(hyp.EPOCHS):
        reset, data = sm.next_epoch_waterfall() #this gets you the entire cow, so to speak
        label = sm.get_label()
        label = np.reshape(label, [1, 1])
        data = np.reshape(data, [hyp.FOOTPRINT,1,6])
        loss_ = 0

        if reset:  # this allows for hidden states to reset after the training set loops back around
            next_state = np.zeros(shape=[2,1,hyp.cell_dim])

        next_state, output_, loss_, summary, _ = sess.run([curr_state, output, loss, summary_op, optimizer],
                                                          feed_dict = {inputs:data, Y:label, init_state:next_state})

        logger.writerow([loss_])

        if epoch % 50 == 0: #50
            writer.add_summary(summary, global_step=epoch)
            print("I finished epoch ", epoch, " out of ", hyp.EPOCHS, " epochs")
            print("The absolute value loss for this sample is ", loss_)
            print("predicted number: ", output_, ", real number: ", label)

        if epoch % 50 == 0 and epoch > hyp.EPOCHS-(50*hyp.FINALJUMP):
            test_local_ = open("evaluation/continuous/currweather/RELU/models/" + str(epoch) + ".csv", 'w')
            test_local = csv.writer(test_local_, lineterminator='\n')

            saver.save(sess, "evaluation/continuous/currweather/RELU/GRAPHS/LSTMweatherRELU", global_step=epoch)

            RMS_loss = 0.0
            next_state_test = np.zeros(shape=[2, 1, hyp.cell_dim])
            carrier = ["true_values", "predicted_values", "abs_error"]
            test_local.writerow(carrier)
            sm.reset_test_counter()
            for test in range(hyp.Info.TEST_SIZE):  # this will be replaced later
                data = sm.next_epoch_test_waterfall()
                label_ = sm.get_label()
                label = np.reshape(label_, [1, 1])
                data = np.reshape(data, [hyp.FOOTPRINT, 1, 6])

                next_state_test, output_, loss_ = sess.run([pass_back_state, output, loss],
                                                           # why passback? Because we only shift by one!
                                                           feed_dict={inputs: data, Y: label, init_state: next_state_test})
                RMS_loss += loss_
                carrier = [label_, output_[0][0], loss_]
                test_local.writerow(carrier)
            RMS_loss = RMS_loss / hyp.Info.TEST_SIZE
            print("doing some rapid tests: this one had loss of " + str(RMS_loss))
            test_local_.close()

        if epoch % 2000 == 0 and epoch > 498:
            saver.save(sess, "evaluation/continuous/currweather/RELU/models/LSTMweatherRELU", global_step=epoch)
            print("---------------------saved model-------------------------")
            sm.create_validation_set()
            RMS_loss = 0.0
            next_state_valid = np.zeros(shape=[2, 1, hyp.cell_dim])
            for i in range(hyp.VALIDATION_NUMBER):
                data = sm.next_epoch_valid_waterfall()
                label_ = sm.get_label()
                label = np.reshape(label_, [1, 1])
                data = np.reshape(data, [hyp.FOOTPRINT, 1, 6])

                next_state_valid, loss_ = sess.run([pass_back_state, loss], #why passback? Because we only shift by one!
                                               feed_dict = {inputs:data, Y:label, init_state:next_state_valid})
                RMS_loss += loss_
            sm.clear_valid_counter()

            RMS_loss = RMS_loss / hyp.VALIDATION_NUMBER
            print("validation: RMS loss is ", RMS_loss)
            validation_logger.writerow([epoch, RMS_loss])




    RMS_loss = 0.0
    next_state_test = np.zeros(shape=[2, 1, hyp.cell_dim])
    carrier = ["true_values", "predicted_values", "abs_error"]
    test_logger.writerow(carrier)
    sm.reset_test_counter()
    for test in range(hyp.Info.TEST_SIZE):  # this will be replaced later
        data = sm.next_epoch_test_waterfall()
        label_ = sm.get_label()
        label = np.reshape(label_, [1, 1])
        data = np.reshape(data, [hyp.FOOTPRINT, 1, 6])

        next_state_test, output_, loss_ = sess.run([pass_back_state, output, loss],  # why passback? Because we only shift by one!
                                     feed_dict={inputs: data, Y: label, init_state: next_state_test})
        RMS_loss += loss_
        carrier = [label_, output_[0][0], loss_]
        test_logger.writerow(carrier)
    RMS_loss = RMS_loss / hyp.Info.TEST_SIZE
    print("test: rms loss is ", RMS_loss)
    test_logger.writerow(["final adaptive loss average", RMS_loss])