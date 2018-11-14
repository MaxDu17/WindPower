"""Maximilian Du 7-2-18
LSTM implementation with wind data set
Version 6 changes:
-trying to concatenate the keep and forget gates into one (see ppt)
-still holding off on validation for now
"""
import tensorflow as tf
import numpy as np
from pipeline import SetMaker
from pipeline import Hyperparameters
from pipeline import My_Loss
import os
import csv

sm = SetMaker()
hyp = Hyperparameters()
ml = My_Loss()

#constructing the big weight now
with tf.name_scope("weights_and_biases"):
    W_Forget_and_Input = tf.Variable(tf.random_normal(shape = [hyp.hidden_dim +1,hyp.cell_dim]), name = "forget_and_input_weight") #note that forget_and_input actually works for forget, and the input is the inverse
    W_Output = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1,hyp.cell_dim]), name="output_weight")
    W_Gate = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1, hyp.cell_dim]), name="gate_weight")

    W_Hidden_to_Out = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim,1]), name = "outwards_propagating_weight")

    B_Forget_and_Input = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name = "forget_and_input_bias")
    B_Output = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="output_bias")
    B_Gate = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="gate_bias")
    B_Hidden_to_Out = tf.Variable(tf.zeros(shape=[1,1]), name = "outwards_propagating_bias")

with tf.name_scope("placeholders"):
    X = tf.placeholder(shape = [1,1], dtype =  tf.float32, name = "input_placeholder") #waits for the prompt
    Y = tf.placeholder(shape = [1,1], dtype = tf.float32, name = "label") #not used until the last cycle
    H_last = tf.placeholder(shape = [1,hyp.hidden_dim], dtype = tf.float32, name = "last_hidden") #last hidden state (aka the "output")
    C_last = tf.placeholder(shape= [1,hyp.cell_dim], dtype = tf.float32, name = "last_cell") #last cell state

with tf.name_scope("to_gates"):
    concat_input = tf.concat([X, H_last], axis = 1, name = "input_concat") #concatenates the inputs to one vector
    forget_gate = tf.add(tf.matmul(concat_input, W_Forget_and_Input, name = "f_w_m"),B_Forget_and_Input, name = "f_b_a") #decides which to drop from cell

    gate_gate = tf.add(tf.matmul(concat_input, W_Gate, name = "g_w_m"), B_Gate, name = "g_b_a") #decides which things to change in cell state
    output_gate = tf.add(tf.matmul(concat_input, W_Output, name="o_w_m"), B_Output, name="o_b_a")

with tf.name_scope("non-linearity"): #makes the gates into what they should be
    forget_gate = tf.sigmoid(forget_gate, name = "sigmoid_forget")

    forget_gate_negated = tf.scalar_mul(-1, forget_gate) #this has to be here because it is after the nonlin
    input_gate = tf.add(tf.ones([1, hyp.cell_dim]), forget_gate_negated, name="making_input_gate")
    input_gate = tf.sigmoid(input_gate, name="sigmoid_input")

    gate_gate = tf.tanh(gate_gate, name = "tanh_gate")
    output_gate = tf.sigmoid(output_gate, name="sigmoid_output")
with tf.name_scope("forget_gate"): #forget gate values and propagate

    current_cell = tf.multiply(forget_gate, C_last, name = "forget_gating")

with tf.name_scope("suggestion_node"): #suggestion gate
    suggestion_box = tf.multiply(input_gate, gate_gate, name = "input_determiner")
    current_cell = tf.add(suggestion_box, current_cell, name = "input_and_gate_gating")

with tf.name_scope("output_gate"): #output gate values to hidden
    current_cell = tf.tanh(current_cell, name = "cell_squashing") #squashing the current cell, branching off now. Note the underscore, means saving a copy.
    current_hidden = tf.multiply(output_gate, current_cell, name="next_hidden") #we are making the hidden by element-wise multiply of the squashed states

    raw_output = tf.add(tf.matmul(current_hidden, W_Hidden_to_Out, name = "WHTO_w_m"), B_Hidden_to_Out, name = "BHTO_b_a") #now, we are propagating outwards

    output = tf.nn.relu(raw_output, name = "output") #makes sure it is not zero.

with tf.name_scope("loss"):
    loss = tf.square(tf.subtract(output, Y))
    loss = tf.reduce_sum(loss)

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=hyp.LEARNING_RATE).minimize(loss)

with tf.name_scope("summaries_and_saver"):
    tf.summary.histogram("W_Forget", W_Forget_and_Input)
    tf.summary.histogram("W_Output", W_Output)
    tf.summary.histogram("W_Gate", W_Gate)
    tf.summary.histogram("W_Hidden_to_Out", W_Hidden_to_Out)

    tf.summary.histogram("Forget", forget_gate)
    tf.summary.histogram("Input", input_gate)
    tf.summary.histogram("Output", output_gate)
    tf.summary.histogram("Gate", gate_gate)

    tf.summary.histogram("Cell_State", current_cell)

    tf.summary.histogram("B_Forget", B_Forget_and_Input)
    tf.summary.histogram("B_Output", B_Output)
    tf.summary.histogram("B_Gate", B_Gate)
    tf.summary.histogram("B_Hidden_to_Out", B_Hidden_to_Out)

    tf.summary.scalar("Loss", loss)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('2012/v6/models/'))
    if ckpt and ckpt.model_checkpoint_path:
        query = input("checkpoint detected! Would you like to restore from <" + ckpt.model_checkpoint_path + "> ?(y or n)\n")
        if query == 'y':
            saver.restore(sess, ckpt.model_checkpoint_path)
            if np.sum(B_Forget_and_Input.eval()) != 0:
                print("session restored!")
        else:
            print("session discarded!")


    sm.create_training_set()
    log_loss = open("2012/v6/GRAPHS/LOSS.csv", "w")
    validation = open("2012/v6/GRAPHS/VALIDATION.csv", "w")
    test = open("2012/v6/GRAPHS/TEST.csv", "w")

    logger = csv.writer(log_loss, lineterminator="\n")
    validation_logger = csv.writer(validation, lineterminator="\n")
    test_logger = csv.writer(test, lineterminator="\n")

    sess.run(tf.global_variables_initializer())

    tf.train.write_graph(sess.graph_def, '2012/v6/GRAPHS/', 'graph.pbtxt')
    writer = tf.summary.FileWriter("2012/v6/GRAPHS/", sess.graph)

    summary = None
    next_cell = np.zeros(shape=[1, hyp.cell_dim])
    next_hidd = np.zeros(shape=[1, hyp.hidden_dim])

    for epoch in range(hyp.EPOCHS):

        sm.next_epoch()
        label = sm.get_label()
        label = np.reshape(label, [1, 1])
        loss_ = 0
        for counter in range(hyp.FOOTPRINT):
            data = sm.next_sample()
            data = np.reshape(data, [1,1])
            if counter < hyp.FOOTPRINT-1:
                next_cell, next_hidd = sess.run([current_cell, current_hidden],
                                                feed_dict= {X:data, H_last:next_hidd, C_last:next_cell})
            else:
                next_cell, next_hidd, output_, loss_, summary, _ = sess.run([current_cell, current_hidden, output, loss, summary_op, optimizer],
                                                feed_dict={X:data, Y:label,  H_last:next_hidd, C_last:next_cell})

        logger.writerow([loss_])

        if epoch%10 == 0:
            writer.add_summary(summary, global_step=epoch)
            print("I finished epoch ", epoch, " out of ", hyp.EPOCHS, " epochs")
            print("The absolute value loss for this sample is ", np.sqrt(loss_))
            print("predicted number: ", output_, ", real number: ", label)

        if epoch%2000 == 0 and epoch>498:
            saver.save(sess, "2012/v6/models/LSTMv6", global_step=epoch)
            print("saved model")

            next_cell_hold = next_cell
            next_hidd_hold = next_hidd
            sm.create_validation_set()
            average_rms_loss = 0.0
            next_cell = np.zeros(shape=[1, hyp.cell_dim])
            next_hidd = np.zeros(shape=[1, hyp.hidden_dim])
            for i in range(hyp.VALIDATION_NUMBER):

                sm.next_epoch_valid()
                label_ = sm.get_label()
                label = np.reshape(label_, [1, 1])
                # this gets each 10th
                for counter in range(hyp.FOOTPRINT):
                    data = sm.next_sample()
                    data = np.reshape(data, [1, 1])
                    if counter < hyp.FOOTPRINT - 1:

                        next_cell, next_hidd = sess.run([current_cell, current_hidden],
                                                        feed_dict={X: data, H_last: next_hidd, C_last: next_cell})
                        if counter == 0:
                            hidden_saver = next_hidd  # saves THIS state for the next round
                            cell_saver = next_cell
                    else:
                        next_cell, next_hidd, output_, loss_ = sess.run(
                            [current_cell, current_hidden, output, loss],
                            feed_dict={X: data, Y: label, H_last: next_hidd, C_last: next_cell})


                next_cell = cell_saver
                next_hidden = hidden_saver
                average_rms_loss += np.sqrt(loss_)
                sm.clear_valid_counter()

            average_rms_loss = average_rms_loss/hyp.VALIDATION_NUMBER
            print("validation: RMS loss is ", average_rms_loss)
            validation_logger.writerow([epoch, average_rms_loss])

            next_cell = next_cell_hold
            next_hidd = next_hidd_hold
    RMS_loss = 0.0
    next_cell = np.zeros(shape=[1, hyp.cell_dim])
    next_hidd = np.zeros(shape=[1, hyp.hidden_dim])
    for test in range(hyp.Info.TEST_SIZE): #this will be replaced later

        sm.next_epoch_test_single_shift()
        label_ = sm.get_label()
        label = np.reshape(label_, [1, 1])
        # this gets each 10th
        for counter in range(hyp.FOOTPRINT):
            data = sm.next_sample()
            data = np.reshape(data, [1, 1])
            if counter < hyp.FOOTPRINT - 1:

                next_cell, next_hidd = sess.run([current_cell, current_hidden],
                                                feed_dict={X: data, H_last: next_hidd, C_last: next_cell})
                if counter == 0:
                    hidden_saver = next_hidd  # saves THIS state for the next round
                    cell_saver = next_cell
            else:
                next_cell, next_hidd, output_, loss_ = sess.run(
                    [current_cell, current_hidden, output, loss],
                    feed_dict={X: data, Y: label, H_last: next_hidd, C_last: next_cell})

                carrier = [label_, output_[0][0], np.sqrt(loss_)]
                test_logger.writerow(carrier)
        next_cell = cell_saver
        next_hidden = hidden_saver

    RMS_loss = RMS_loss / hyp.Info.TEST_SIZE
    print("test: rms loss is ", RMS_loss)
    test_logger.writerow(["final adaptive loss average", RMS_loss])