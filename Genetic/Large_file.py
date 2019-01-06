import subprocess
import random
import csv
import tensorflow as tf
from pipeline.dataset_maker import SetMaker
import numpy as np
POPULATION_SIZE = 10
TRAINING_EPOCHS = 500 #used to be 500
TEST_SIZE = 200
ACTIVE_HYP = 3
CROSSOVER = 3
GENETIC_EPOCHS = 20
MUTATION_RATE = 0.3


genetic_matrix = []
data_dict = {}
subprocess_array = []

def graph(hyperparameters, sess):
    footprint = hyperparameters[0]
    learning_rate = hyperparameters[1]
    cell_dim = hidden_dim = hyperparameters[2]
    #hidden_dim = hyperparameters[3]
    epochs = hyperparameters[3]  # just a data issue. No data is being destroyed here. I'm just changing it to a compatible type
    test_size = hyperparameters[4]
    SERIAL_NUMBER = hyperparameters[5] # this is for telling which instance this is

    sm = SetMaker(footprint)
    with tf.name_scope("weights_and_biases"):
        W_Forget = tf.Variable(tf.random_normal(shape=[hidden_dim + 1, cell_dim]), name="forget_weight")
        #W_Forget = tf.Variable(tf.random_normal(shape=[cell_dim, hidden_dim + 1]), name="forget_weight")
        W_Output = tf.Variable(tf.random_normal(shape=[hidden_dim + 1, cell_dim]), name="output_weight")
        W_Gate = tf.Variable(tf.random_normal(shape=[hidden_dim + 1, cell_dim]), name="gate_weight")
        W_Input = tf.Variable(tf.random_normal(shape=[hidden_dim + 1, cell_dim]), name="input_weight")
        W_Hidden_to_Out = tf.Variable(tf.random_normal(shape=[hidden_dim, 1]), name="outwards_propagating_weight")

        B_Forget = tf.Variable(tf.zeros(shape=[1, cell_dim]), name="forget_bias")
        B_Output = tf.Variable(tf.zeros(shape=[1, cell_dim]), name="output_bias")
        B_Gate = tf.Variable(tf.zeros(shape=[1, cell_dim]), name="gate_bias")
        B_Input = tf.Variable(tf.zeros(shape=[1, cell_dim]), name="input_bias")
        B_Hidden_to_Out = tf.Variable(tf.zeros(shape=[1, 1]), name="outwards_propagating_bias")

    with tf.name_scope("placeholders"):
        Y = tf.placeholder(shape=[1, 1], dtype=tf.float32, name="label")  # not used until the last cycle
        init_state = tf.placeholder(shape=[2, 1, cell_dim], dtype=tf.float32, name="initial_states") #problem here
        #init_state_cell = tf.placeholder(shape=[1, 1, cell_dim], dtype=tf.float32, name="initial_states_cell")
        #init_state_hidden = tf.placeholder(shape=[1, 1, hidden_dim], dtype=tf.float32, name="initial_states_hidden")
        inputs = tf.placeholder(shape=[footprint, 1, 1], dtype=tf.float32, name="input_data")


    def step(last_state, X):
        with tf.name_scope("to_gates"):
            C_last, H_last = tf.unstack(last_state)
            concat_input = tf.concat([X, H_last], axis=1,
                                     name="input_concat")  # concatenates the inputs to one vector
            forget_gate = tf.add(tf.matmul(concat_input, W_Forget, name="f_w_m"), B_Forget,
                                 name="f_b_a")  # decides which to drop from cell
            output_gate = tf.add(tf.matmul(concat_input, W_Output, name="o_w_m"), B_Output,
                                 name="o_b_a")  # decides which to reveal to next_hidd/output
            gate_gate = tf.add(tf.matmul(concat_input, W_Gate, name="g_w_m"), B_Gate,
                               name="g_b_a")  # decides which things to change in cell state
            input_gate = tf.add(tf.matmul(concat_input, W_Input, name="i_w_m"), B_Input,
                                name="i_b_a")  # decides which of the changes to accept

        with tf.name_scope("non-linearity"):  # makes the gates into what they should be
            forget_gate = tf.sigmoid(forget_gate, name="sigmoid_forget")
            output_gate = tf.sigmoid(output_gate, name="sigmoid_output")
            input_gate = tf.sigmoid(input_gate, name="sigmoid_input")
            gate_gate = tf.tanh(gate_gate, name="tanh_gate")

        with tf.name_scope("forget_gate"):  # forget gate values and propagate
            current_cell = tf.multiply(forget_gate, C_last, name="forget_gating")

        with tf.name_scope("suggestion_node"):  # suggestion gate
            suggestion_box = tf.multiply(input_gate, gate_gate, name="input_determiner")
            current_cell = tf.add(suggestion_box, current_cell, name="input_and_gate_gating")

        with tf.name_scope("output_gate"):  # output gate values to hidden
            current_cell = tf.tanh(current_cell, name="output_presquashing")
            current_hidden = tf.multiply(output_gate, current_cell, name="next_hidden")
            states = tf.stack([current_cell, current_hidden])
        return states

    with tf.name_scope("forward_roll"):
        #init_state = np.stack([init_state_cell, init_state_hidden])
        states_list = tf.scan(fn=step, elems=inputs, initializer=init_state, name="scan")
        curr_state = states_list[-1]
        pass_back_state = tf.add([0.0], states_list[0], name="pass_back_state")

    with tf.name_scope("prediction"):
        _, current_hidden = tf.unstack(curr_state)
        raw_output = tf.add(tf.matmul(current_hidden, W_Hidden_to_Out, name="WHTO_w_m"), B_Hidden_to_Out,
                            name="BHTO_b_a")
        output = tf.nn.relu(raw_output, name="output")

    with tf.name_scope("loss"):
        loss = tf.square(tf.subtract(output, Y))
        loss = tf.reshape(loss, [], name="loss")

    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    #with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    sm.create_training_set()
    summary = None  # this is just because it was used before

    #next_state_cell = np.zeros(shape=[1, 1, cell_dim])
    #next_state_hidd = np.zeros(shape=[1, 1, hidden_dim])
    #next_state = np.stack([next_state_cell, next_state_hidd])
    next_state = np.zeros(shape=[2, 1, cell_dim])

    for epoch in range(epochs):
        reset, data = sm.next_epoch_waterfall()  # this gets you the entire cow, so to speak
        label = sm.get_label()
        label = np.reshape(label, [1, 1])
        data = np.reshape(data, [footprint, 1, 1])
        loss_ = 0

        if reset:  # this allows for hidden states to reset after the training set loops back around
            # next_state_cell = np.zeros(shape=[1, 1, cell_dim])
            # next_state_hidd = np.zeros(shape=[1, 1, hidden_dim])
            # next_state = np.stack([next_state_cell, next_state_hidd])
            next_state = np.zeros(shape=[2,1,cell_dim])

        next_state, loss_, _ = sess.run([curr_state, loss, optimizer],
                                        feed_dict={inputs: data, Y: label, init_state: next_state})

    RMS_loss = 0.0
    next_state = np.zeros(shape=[2, 1, cell_dim])
    # print(np.shape(next_state))
    for test in range(test_size):  # this will be replaced later

        data = sm.next_epoch_test_waterfall()
        label_ = sm.get_label()
        label = np.reshape(label_, [1, 1])
        data = np.reshape(data, [footprint, 1, 1])

        next_state, output_, loss_ = sess.run([pass_back_state, output, loss],
                                              # why passback? Because we only shift by one!
                                              feed_dict={inputs: data, Y: label, init_state: next_state})
        RMS_loss += np.sqrt(loss_)
        # carrier = [label_, output_[0][0], np.sqrt(loss_)]
        # test_logger.writerow(carrier)
    RMS_loss = RMS_loss / test_size
    print("test for " + str(SERIAL_NUMBER) + ": rms loss is ", RMS_loss)
    return RMS_loss

def sort_second(val):
    return val[1]

def is_mutate():
    if(random.random() > MUTATION_RATE):
        return True
    else:
        return False

def parent_picker():
    result = random.random()
    if result > 0.5:
        return 1
    else:
        return 2

def mutate(value):
    type_ = type(value).__name__
    if type_ == "int":
        result = mutate_int(value)
    elif type_ == "float":
        result = mutate_float(value)
    else:
        raise ValueError("The type was not caught")

    return result

def mutate_int(value):
    mutation = is_mutate() # this checks if we are mutating
    if mutation:  # if we are actually mutating
        random_result = random.randint(1, 2)  # we do a coin flip
        if random_result == 1:  # this arbitrary case means we increment
            value += 1
        elif random_result == 2:  # this arbitrary case means we decrement
            value -= 1

        if value == 0: #this is a "dumb" way to prevent zero error
            value += 1

    return value  # returns the modified value

def mutate_float(value):
    mutation = is_mutate()
    if mutation:
        random_shift = random.uniform(-0.002, 0.002)
        #print(random_shift)
        value += random_shift
    return value


def cross_over(array_1, array_2):
    scratch_list = list()
    child_list = list()
    for i in range(POPULATION_SIZE - 2): #minus 2 b/c the parents will stay too
        for i in range(CROSSOVER):
            parent = parent_picker()
            if parent==1:
                scratch_list.append(mutate(array_1[i]))
            else:
                scratch_list.append(mutate(array_2[i]))
        child_list.append(scratch_list)
        scratch_list = list()#we're resetting this
    child_list.append(array_1)
    child_list.append(array_2)
    return child_list



with tf.Session() as sess:
    first = True
    for k in range(GENETIC_EPOCHS):
        print("This is epoch: " + str(k))
        results = list()
        for i in range(POPULATION_SIZE):

            if first:
                learning_rate = round(random.randrange(1, 20) * 0.0005, 6)
                footprint = int(random.randint(5, 15))
                cell_hidden_dim = random.randint(10, 100)
                genetic_matrix = [footprint, learning_rate, cell_hidden_dim, TRAINING_EPOCHS, TEST_SIZE, i]
                results.append([genetic_matrix, graph(genetic_matrix, sess)])

            else:
                children[i].append(TRAINING_EPOCHS)
                children[i].append(TEST_SIZE)
                children[i].append(i)
                results.append([children[i], graph(children[i], sess)])


        results.sort(key = sort_second)
        results = [k[0] for k in results] #removes the error. This is no longer needed
        results = [k[0:3] for k in results] #removes the serial number. This is no longer needed
        children = cross_over(results[0], results[1]) #this should g et the hyperparameters
        first = False

        print(children)
        print("The kept parents are: " + str(results[0:2]))

    k = open("best.csv", "w")
    best_writer = csv.writer(k, lineterminator = "\n")
    best_writer.writerow(results[0:2])
