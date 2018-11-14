"""Maximilian Du 6-28-18
LSTM implementation with wind data set"""
import tensorflow as tf
import numpy as np
from pipeline import SetMaker
from pipeline import Hyperparameters
import os
import csv

sm = SetMaker()
hyp = Hyperparameters()

#constructing the big weight now
with tf.name_scope("weights_and_biases"):
    W_Forget = tf.Variable(tf.random_normal(shape = [hyp.hidden_dim + 1,hyp.cell_dim]), name = "forget_weight")
    W_Output = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1,hyp.cell_dim]), name="output_weight")
    W_Gate = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1, hyp.cell_dim]), name="gate_weight")
    W_Input = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1, hyp.cell_dim]), name="input_weight")
    W_Hidden_to_Out = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim,1]), name = "outwards_propagating_weight")

    B_Forget = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name = "forget_bias")
    B_Output = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="output_bias")
    B_Gate = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="gate_bias")
    B_Input = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="input_bias")
    B_Hidden_to_Out = tf.Variable(tf.zeros(shape=[1,1]), name = "outwards_propagating_bias")

with tf.name_scope("placeholders"):
    X = tf.placeholder(shape = [1,1], dtype =  tf.float32, name = "input_placeholder") #waits for the prompt
    Y = tf.placeholder(shape = [1,1], dtype = tf.float32, name = "label") #not used until the last cycle
    H_last = tf.placeholder(shape = [1,hyp.hidden_dim], dtype = tf.float32, name = "last_hidden") #last hidden state (aka the "output")
    C_last = tf.placeholder(shape= [1,hyp.cell_dim], dtype = tf.float32, name = "last_cell") #last cell state

with tf.name_scope("to_gates"):
    X = tf.reshape(X, shape = [1,1])
    H_last_ = tf.reshape(H_last, shape = [hyp.hidden_dim,1])
    concat_input = tf.concat([X, H_last_], axis = 0, name = "input_concat") #concatenates the inputs to one vector
    concat_input = tf.transpose(concat_input)
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
    current_hidden = tf.multiply(output_gate, current_cell, name = "next_hidden")
    output = tf.add(tf.matmul(current_hidden, W_Hidden_to_Out, name = "WHTO_w_m"), B_Hidden_to_Out, name = "BHTO_b_a")

with tf.name_scope("loss"):
    loss = tf.square(tf.subtract(output, Y))
    loss = tf.reshape(loss, [])

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=hyp.LEARNING_RATE).minimize(loss)

with tf.name_scope("summaries_and_saver"):
    tf.summary.histogram("W_Forget", W_Forget)
    tf.summary.histogram("W_Input", W_Input)
    tf.summary.histogram("W_Output", W_Output)
    tf.summary.histogram("W_Gate", W_Gate)

    tf.summary.histogram("Cell_State", current_cell)

    tf.summary.histogram("B_Forget", B_Forget)
    tf.summary.histogram("B_Input", B_Input)
    tf.summary.histogram("B_Output", B_Output)
    tf.summary.histogram("B_Gate", B_Gate)

    tf.summary.scalar("Loss", loss)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('v1/models/'))
    print(ckpt)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    sm.create_training_set()
    log_loss = open("v1/GRAPHS/LOSS.csv", "w")
    validation = open("v1/GRAPHS/VALIDATION.csv", "w")
    test = open("v1/GRAPHS/TEST.csv", "w")

    logger = csv.writer(log_loss, lineterminator="\n")
    validation_logger = csv.writer(validation, lineterminator="\n")
    test_logger = csv.writer(test, lineterminator="\n")

    sess.run(tf.global_variables_initializer())

    tf.train.write_graph(sess.graph_def, 'v1/GRAPHS/', 'graph.pbtxt')
    writer = tf.summary.FileWriter("v1/GRAPHS/", sess.graph)
    summary = None

    for epoch in range(hyp.EPOCHS):
        sm.next_epoch()
        label = sm.get_label()
        label = np.reshape(label, [1, 1])
        next_cell = np.zeros(shape=[1, hyp.cell_dim])
        next_hidd = np.zeros(shape=[1, hyp.hidden_dim])
        loss_ = 0
        for counter in range(hyp.FOOTPRINT):
            data = sm.next_sample()
            data = np.reshape(data, [1,1])
            if counter < hyp.FOOTPRINT-1:
                next_cell, next_hidd = sess.run([current_cell, current_hidden],
                                                feed_dict= {X:data, H_last:next_hidd, C_last:next_cell})
            else:
                next_cell, output_, loss_, summary, _ = sess.run([current_cell, output, loss, summary_op, optimizer],
                                                feed_dict={X:data, Y:label,  H_last:next_hidd, C_last:next_cell})


        writer.add_summary(summary, global_step=epoch)
        print("I finished epoch ", epoch, " out of ", hyp.EPOCHS, " epochs")
        print("The squared loss for this sample is ", loss_)
        logger.writerow([loss_])

        if epoch%10 == 0:
            print("predicted number: ", output_, ", real number: ", label)

        if epoch%500 == 0:
            saver.save(sess, "v1/models/LSTMv1", global_step=epoch)
            print("saved model")

        if epoch%1000 == 0 and epoch>498:
            sm.create_validation_set()
            average_sq_loss = 0.0
            for i in range(hyp.VALIDATION_NUMBER):
                sm.next_epoch_valid()
                label = sm.get_label()
                label = np.reshape(label, [1, 1])
                next_cell = np.zeros(shape=[1, hyp.cell_dim])
                next_hidd = np.zeros(shape=[1, hyp.hidden_dim])

                for counter in range(hyp.FOOTPRINT):
                    data = sm.next_sample()
                    data = np.reshape(data, [1, 1])
                    if counter < hyp.FOOTPRINT-1:
                        next_cell, next_hidd = sess.run([current_cell, current_hidden],
                                                        feed_dict={X: data, H_last: next_hidd, C_last: next_cell})
                    else:
                        output_, loss_= sess.run(
                            [output, loss],
                            feed_dict={X: data, Y:label, H_last: next_hidd, C_last: next_cell})
                        average_sq_loss += loss_
                sm.clear_valid_counter()

            average_sq_loss = average_sq_loss/hyp.VALIDATION_NUMBER
            print("validation: average square loss is ", average_sq_loss)
            validation_logger.writerow([epoch, average_sq_loss])


    average_sq_loss = 0.0
    for test in range(hyp.Info.TEST_SIZE): #this will be replaced later

        sm.next_epoch_test()
        label = sm.get_label()
        label = np.reshape(label, [1, 1])
        next_cell = np.zeros(shape=[1, hyp.cell_dim])
        next_hidd = np.zeros(shape=[1, hyp.hidden_dim])

        for counter in range(hyp.FOOTPRINT):
            data = sm.next_sample()
            data = np.reshape(data, [1, 1])
            if counter < hyp.FOOTPRINT-1:
                next_cell, next_hidd = sess.run([current_cell, current_hidden],
                                                feed_dict={X: data, H_last: next_hidd, C_last: next_cell})
            else:
                output_, loss_ = sess.run(
                    [output, loss],
                    feed_dict={X: data, Y:label, H_last: next_hidd, C_last: next_cell})
                average_sq_loss += loss_
                test_logger.writerow([loss_])

    average_sq_loss = average_sq_loss / hyp.Info.TEST_SIZE
    print("test: average square loss is ", average_sq_loss)
    test_logger.writerow(["this is the final average squared loss", average_sq_loss])

