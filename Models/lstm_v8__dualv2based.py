"""Maximilian Du 7-16-18
LSTM implementation with wind data set
Version 8 changes:
multi-layer LSTM!
"""
import tensorflow as tf
import numpy as np
from pipeline import SetMaker
from pipeline import Hyperparameters
from pipeline import Model2
import os
import csv

sm = SetMaker()
hyp = Hyperparameters()

layer_1 = Model2()
layer_2 = Model2()

with tf.name_scope("placeholders"):
    Y = tf.placeholder(shape=[1, 1], dtype=tf.float32, name="label")  # not used until the last cycle
    init_state_1 = tf.placeholder(shape = [2,1,hyp.cell_dim], dtype = tf.float32, name = "initial_states_1")
    init_state_2 = tf.placeholder(shape = [2,1,hyp.cell_dim], dtype = tf.float32, name = "initial_states_2")
    inputs = tf.placeholder(shape = [hyp.FOOTPRINT,1,1], dtype = tf.float32,  name = "input_data")

with tf.name_scope("aux_weights_and_biases"):
    W_Hidden_to_Out_1 = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim, 1]),
                                  name="outwards_propagating_weight_1")
    W_Hidden_to_Out_2 = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim, 1]),
                                  name="outwards_propagating_weight_2")
    B_Hidden_to_Out_1 = tf.Variable(tf.zeros(shape=[1, 1]), name="outwards_propagating_bias_1")
    B_Hidden_to_Out_2 = tf.Variable(tf.zeros(shape=[1, 1]), name="outwards_propagating_bias_2")

with tf.name_scope("layer_1_propagation"):
    states_1 = layer_1.create_graph(layer_number = 1, inputs = inputs, init_state = init_state_1)
    curr_state_1 = states_1[-1]
    pass_back_state_1 = tf.add([0.0], states_1[0], name = "pass_back_state_1")
    input_2 = list()
    for i in range(hyp.FOOTPRINT):
        _, current_hidden = tf.unstack(states_1[i])
        current_input_2 = tf.add(tf.matmul(current_hidden, W_Hidden_to_Out_1, name="WHTO_w_m"), B_Hidden_to_Out_1, name="BHTO_b_a")
        input_2.append(current_input_2)
    input_2 = tf.reshape(input_2, [hyp.FOOTPRINT,1,1])


with tf.name_scope("layer_2_propagation"):
    states_2 = layer_2.create_graph(layer_number = 2, inputs = input_2, init_state = init_state_2)
    curr_state_2 = states_2[-1]
    pass_back_state_2 = tf.add([0.0], states_2[0], name = "pass_back_state_2")
    _, current_hidden_2 = tf.unstack(curr_state_2)
    raw_output = tf.add(tf.matmul(current_hidden_2, W_Hidden_to_Out_2, name="WHTO_w_m_"), B_Hidden_to_Out_2, name="BHTO_b_a")
    output = tf.nn.relu(raw_output, name="output")

with tf.name_scope("loss"):
    loss = tf.square(tf.subtract(output, Y))
    loss = tf.reshape(loss, [], name = "loss")

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=hyp.LEARNING_RATE).minimize(loss)

with tf.name_scope("summaries_and_saver"):
    tf.summary.scalar("Loss", loss)
    tf.summary.histogram("W_Hidden_to_Out", W_Hidden_to_Out_1)
    tf.summary.histogram("B_Hidden_to_Out", B_Hidden_to_Out_1)
    tf.summary.histogram("W_Hidden_to_Out", W_Hidden_to_Out_2)
    tf.summary.histogram("B_Hidden_to_Out", B_Hidden_to_Out_2)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('2012/v8/models/'))
    if ckpt and ckpt.model_checkpoint_path:
        query = input(
            "checkpoint detected! Would you like to restore from <" + ckpt.model_checkpoint_path + "> ?(y or n)\n")
        if query == 'y':
            saver.restore(sess, ckpt.model_checkpoint_path)
            if np.sum(layer_1.B_Forget.eval()) != 0:
                print("session restored!")
        else:
            print("Starting from scratch!")

    log_loss = open("2012/v8/GRAPHS/LOSS.csv", "w")
    validation = open("2012/v8/GRAPHS/VALIDATION.csv", "w")
    test = open("2012/v8/GRAPHS/TEST.csv", "w")
    logger = csv.writer(log_loss, lineterminator="\n")
    validation_logger = csv.writer(validation, lineterminator="\n")
    test_logger = csv.writer(test, lineterminator="\n")

    sm.create_training_set()

    tf.train.write_graph(sess.graph_def, '2012/v8/GRAPHS/', 'graph.pbtxt')
    writer = tf.summary.FileWriter("2012/v8/GRAPHS/", sess.graph)

    summary = None
    init_state_1_ = init_state_2_ = np.zeros(shape=[2, 1, hyp.cell_dim])

    for epoch in range(hyp.EPOCHS):
        reset, data = sm.next_epoch_waterfall()  # this gets you the entire cow, so to speak
        label = sm.get_label()
        label = np.reshape(label, [1, 1])
        data = np.reshape(data, [hyp.FOOTPRINT, 1, 1])
        loss_ = 0

        if reset:  # this allows for hidden states to reset after the training set loops back around
            init_state_1_ = init_state_2_ = np.zeros(shape=[2, 1, hyp.cell_dim])

        init_state_1_, init_state_2_, output_, loss_, summary, _ = sess.run([curr_state_1, curr_state_2, output, loss, summary_op, optimizer],
                                                                          feed_dict = {Y:label, init_state_1:init_state_1_,
                                                                                       init_state_2:init_state_2_, inputs:data})

        logger.writerow([loss_])


        if epoch%50 == 0:

            writer.add_summary(summary, global_step=epoch)
            print("I finished epoch ", epoch, " out of ", hyp.EPOCHS, " epochs")
            print("The absolute value loss for this sample is ", np.sqrt(loss_))
            print("predicted number: ", output_, ", real number: ", label)

############################################################### validation code here
        if epoch % 2000 == 0 and epoch > 498:
            saver.save(sess, "2012/v8/models/LSTMv8", global_step=epoch)
            print("---------------------saved model-------------------------")

            init_state_1_hold = init_state_1_  # this "pauses" the training that is happening right now.
            init_state_2_hold = init_state_2_
            sm.create_validation_set()
            RMS_loss = 0.0
            init_state_1_ = init_state_2_ = np.zeros(shape=[2, 1, hyp.cell_dim])
            for i in range(hyp.VALIDATION_NUMBER):
                data = sm.next_epoch_valid_waterfall()
                label_ = sm.get_label()
                label = np.reshape(label_, [1, 1])
                data = np.reshape(data, [hyp.FOOTPRINT, 1, 1])

                init_state_1_, init_state_2_, loss_ = sess.run([pass_back_state_1, pass_back_state_2, loss],  # why passback? Because we only shift by one!
                                             feed_dict={Y: label, init_state_1: init_state_1_,
                                                        init_state_2: init_state_2_, inputs: data})
                RMS_loss += np.sqrt(loss_)
            sm.clear_valid_counter()

            RMS_loss = RMS_loss / hyp.VALIDATION_NUMBER
            print("validation: RMS loss is ", RMS_loss)
            validation_logger.writerow([epoch, RMS_loss])

            init_state_1_ = init_state_1_hold
            init_state_2_ = init_state_2_hold  # restoring past point...

    RMS_loss = 0.0
    init_state_1_ = init_state_2_ = np.zeros(shape=[2, 1, hyp.cell_dim])
    print(np.shape(next_state))
    for test in range(hyp.Info.TEST_SIZE):  # this will be replaced later

        data = sm.next_epoch_test_waterfall()
        label_ = sm.get_label()
        label = np.reshape(label_, [1, 1])
        data = np.reshape(data, [hyp.FOOTPRINT, 1, 1])

        init_state_1_, init_state_2_, output_, loss_ = sess.run([pass_back_state_1, pass_back_state_2, output, loss], # why passback? Because we only shift by one!
                                                       feed_dict={Y: label, init_state_1: init_state_1_,
                                                                  init_state_2: init_state_2_, inputs: data})
        RMS_loss += np.sqrt(loss_)
        carrier = [label_, output_[0][0], np.sqrt(loss_)]
        test_logger.writerow(carrier)
    RMS_loss = RMS_loss / hyp.Info.TEST_SIZE
    print("test: rms loss is ", RMS_loss)
    test_logger.writerow(["final adaptive loss average", RMS_loss])