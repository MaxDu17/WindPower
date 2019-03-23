"""Maximilian Du 1-5-19
this takes in a csv and trains the FIRST data point on it
"""
import tensorflow as tf
import numpy as np
from pipeline.dataset_maker_forecast import SetMaker_Forecast
from pipeline.hyperparameters import Hyperparameters
import os
import sys
import csv

NAME = "gru_c_class_FORE" #this is the name of the python file for logging purposes

k = open("../Genetic/" + str(NAME) + "best.csv", "r")

hyp_list =  list(csv.reader(k)) #extracing the first data point from the csv file

FOOTPRINT = int(hyp_list[0][0])
LEARNING_RATE = float(hyp_list[0][1])
hidden_dim = cell_dim = int(hyp_list[0][2])
sm = SetMaker_Forecast(FOOTPRINT)
hyp = Hyperparameters() # this is used later for non-changing hyperparameters
epochs = hyp.EPOCHS_LARGE #this is the epochs setting

if len(sys.argv) > 1:
    epochs = int(sys.argv[1]) #this allows us to provide an arbitrary training size


#constructing the big weight now
with tf.name_scope("weights_and_biases"):
    W_Insertion = tf.Variable(tf.random_normal(shape=[hidden_dim + 21, hidden_dim]), name="insertion_weight")
    W_Input_Forget = tf.Variable(tf.random_normal(shape=[hidden_dim + 21, hidden_dim]), name="Input_Forget_weight")
    W_Suggestion = tf.Variable(tf.random_normal(shape=[hidden_dim + 21, hidden_dim]), name="Suggestion_weight")

    W_Hidden_to_Out = tf.Variable(tf.random_normal(shape=[hidden_dim, 1]), name="outwards_propagating_weight")

    B_Insertion = tf.Variable(tf.zeros(shape=[1, hidden_dim]), name="insertion_bias")
    B_Input_Forget = tf.Variable(tf.zeros(shape=[1, hidden_dim]), name="input_forget_bias")
    B_Suggestion = tf.Variable(tf.zeros(shape=[1, hidden_dim]), name="suggestion_bias")
    B_Hidden_to_Out = tf.Variable(tf.zeros(shape=[1, 1]), name="outwards_propagating_bias")

with tf.name_scope("placeholders"):
    Y = tf.placeholder(shape=[1, 1], dtype=tf.float32, name="label")  # not used until the last cycle
    init_state = tf.placeholder(shape=[1, hidden_dim], dtype=tf.float32, name="initial_states")  # problem here
    inputs = tf.placeholder(shape=[FOOTPRINT, 1, 21], dtype=tf.float32, name="input_data")


def step(last_state, X):
    with tf.name_scope("To_insertion_and_IF_gates"):
        H_last = last_state
        process_rail = tf.concat([X, H_last], axis=1, name="process_rail_concat")  # this makes the process rail
        insertion_gate = tf.add(tf.matmul(process_rail, W_Insertion, name="matmul_insert"), B_Insertion,
                                name="bias_add_insert")  # this makes the gate state
        input_forget_gate = tf.add(tf.matmul(process_rail, W_Input_Forget, name="matmul_IF"), B_Input_Forget,
                                   name="bias_add_IF")  # this makes the other gate state
    with tf.name_scope("To_suggestion_gate"):
        insertion_gate = tf.sigmoid(insertion_gate, name="sigmoid_insertion_gate")  # this makes non-linear stuff
        input_forget_gate = tf.sigmoid(input_forget_gate, name="sigmoid_IF")
        IF_negated = tf.scalar_mul(-1, input_forget_gate)  # this has to be here because it is after the nonlin
        input_gate = tf.add(tf.ones([1, hidden_dim]), IF_negated,
                            name="making_input_gate")  # making the two other gate states
        forget_gate = input_forget_gate  # just to make sytax correct
    with tf.name_scope("To_suggestion_rail_and_node"):
        suggestion_rail_h = tf.multiply(insertion_gate, H_last,
                                        name="suggestion_rail_making")  # this one uses the insertion
        suggestion_rail = tf.concat([X, suggestion_rail_h], axis=1,
                                    name="making_suggestion_rail")  # this one concatenates
        suggestion_gate = tf.add(tf.matmul(suggestion_rail, W_Suggestion, name="matmul_suggestion"), B_Suggestion,
                                 name="bias_add_suggestion")
        suggestion_gate = tf.tanh(suggestion_gate, name="tanh_suggestion")  # makes the suggestion gate
    with tf.name_scope("To_end"):
        suggestion = tf.multiply(suggestion_gate, input_gate,
                                 name="to_suggestion_node")  # this makes the filtered suggestion
        new_hidd = tf.multiply(forget_gate, H_last, name="forget_gating")  # but first, we need to forget some things
        new_hidd = tf.add(new_hidd, suggestion, name="suggestion_adding")  # now, we add to form our next state

    return new_hidd


with tf.name_scope("forward_roll"):  # DO NOT CHANGE THIS NAME!!!
    # init_state = np.stack([init_state_cell, init_state_hidden])
    states_list = tf.scan(fn=step, elems=inputs, initializer=init_state, name="scan")
    curr_state = states_list[-1]
    pass_back_state = tf.add([0.0], states_list[0], name="pass_back_state")

with tf.name_scope("prediction"):  # do NOT CHANGE THIS NAME!!!
    current_hidden = curr_state
    raw_output = tf.add(tf.matmul(current_hidden, W_Hidden_to_Out, name="WHTO_w_m"), B_Hidden_to_Out,
                        name="BHTO_b_a")
    output = tf.nn.relu(raw_output, name="output")

with tf.name_scope("loss"):
    loss = tf.square(tf.subtract(output, Y))
    loss = tf.reshape(loss, [], name="loss")

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

with tf.name_scope("summaries_and_saver"): #these are for tensorboard display
    tf.summary.histogram("W_Insertion", W_Insertion)
    tf.summary.histogram("W_Input_Forget", W_Input_Forget)
    tf.summary.histogram("W_Suggestion", W_Suggestion)
    tf.summary.histogram("W_Hidden_to_Out", W_Hidden_to_Out)

    tf.summary.histogram("B_Insertion", B_Insertion)
    tf.summary.histogram("B_Input_Forget", B_Input_Forget)
    tf.summary.histogram("B_Suggestion", B_Suggestion)
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
            if np.sum(B_Insertion.eval()) != 0: #this checks for restored session
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
    next_state = np.zeros(shape=[1,cell_dim]) #this initializes the initial "next state"

    for epoch in range(epochs):
        reset, data = sm.next_epoch_waterfall() #this gets you the entire data chunk
        label = sm.get_label() #this is the answer key
        label = np.reshape(label, [1, 1]) #reshaping for data transfer
        data = np.reshape(data, [FOOTPRINT,1,21])
        loss_ = 0

        if reset:  # this allows for hidden states to reset after the training set loops back around
            next_state = np.zeros(shape=[1,cell_dim])

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

            saver.save(sess, "../Graphs_and_Results/" + NAME + "/models/GRUGenetic_FORE, global_step=epoch")

            RMS_loss = 0.0
            next_state_test = np.zeros(shape=[1, cell_dim]) #initializations
            carrier = ["true_values", "predicted_values", "abs_error"]
            test_local.writerow(carrier)
            sm.reset_test_counter()

            for test in range(hyp.Info.TEST_SIZE):  # this will be replaced later
                data = sm.next_epoch_test_waterfall()
                label_ = sm.get_label()
                label = np.reshape(label_, [1, 1])
                #data = np.reshape(data, [FOOTPRINT, 1, 6])
                data = np.reshape(data, [FOOTPRINT, 1, 21])
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
            saver.save(sess, "../Graphs_and_Results/" + NAME + "/models/GRUGenetic_FORE", global_step=epoch)
            print("---------------------saved model-------------------------")

            next_state_hold = next_state #this "pauses" the training that is happening right now.
            sm.create_validation_set()
            RMS_loss = 0.0
            next_state = np.zeros(shape=[1, cell_dim])
            for i in range(hyp.VALIDATION_NUMBER):
                data = sm.next_epoch_valid_waterfall()
                label_ = sm.get_label()
                label = np.reshape(label_, [1, 1])
                data = np.reshape(data, [FOOTPRINT, 1, 21])

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
    next_state = np.zeros(shape=[1, cell_dim])
    print(np.shape(next_state))
    sm.reset_test_counter()
    for test in range(hyp.Info.TEST_SIZE):  # this will be replaced later

        data = sm.next_epoch_test_waterfall()
        label_ = sm.get_label()
        label = np.reshape(label_, [1, 1])
        data = np.reshape(data, [FOOTPRINT, 1, 21])

        next_state, output_, loss_ = sess.run([pass_back_state, output, loss],  # why passback? Because we only shift by one!
                                     feed_dict={inputs: data, Y: label, init_state: next_state})
        RMS_loss += np.sqrt(loss_)
        carrier = [label_, output_[0][0], np.sqrt(loss_)]
        test_logger.writerow(carrier)
    RMS_loss = RMS_loss / hyp.Info.TEST_SIZE
    print("test: rms loss is ", RMS_loss)
    test_logger.writerow(["Loss average", RMS_loss])