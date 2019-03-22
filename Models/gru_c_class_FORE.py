import tensorflow as tf
from pipeline.dataset_maker_forecast import SetMaker_Forecast
import numpy as np

class LSTM: #ok, this is a GRU
    def __init__(self):
       print("LSTM created")

    def graph(self, hyperparameters, sess):

        footprint = hyperparameters[0]
        learning_rate = hyperparameters[1]
        hidden_dim = hyperparameters[2]
        #hidden_dim = hyperparameters[3]
        epochs = hyperparameters[3]  # just a data issue. No data is being destroyed here. I'm just changing it to a compatible type
        test_size = hyperparameters[4]
        SERIAL_NUMBER = hyperparameters[5] # this is for telling which instance this is

        sm = SetMaker_Forecast(footprint)
        with tf.name_scope("weights_and_biases"):
            W_Insertion = tf.Variable(tf.random_normal(shape=[hidden_dim + 21, hidden_dim]), name="insertion_weight")
            W_Input_Forget = tf.Variable(tf.random_normal(shape = [hidden_dim + 21, hidden_dim]), name = "Input_Forget_weight")
            W_Suggestion = tf.Variable(tf.random_normal(shape = [hidden_dim + 21, hidden_dim]), name = "Suggestion_weight")

            W_Hidden_to_Out = tf.Variable(tf.random_normal(shape=[hidden_dim, 1]), name="outwards_propagating_weight")

            B_Insertion = tf.Variable(tf.zeros(shape=[1, hidden_dim]), name="insertion_bias")
            B_Input_Forget = tf.Variable(tf.zeros(shape=[1, hidden_dim]), name="input_forget_bias")
            B_Suggestion = tf.Variable(tf.zeros(shape=[1, hidden_dim]), name="suggestion_bias")
            B_Hidden_to_Out = tf.Variable(tf.zeros(shape=[1, 1]), name="outwards_propagating_bias")

        with tf.name_scope("placeholders"):
            Y = tf.placeholder(shape=[1, 1], dtype=tf.float32, name="label")  # not used until the last cycle
            init_state = tf.placeholder(shape=[1, hidden_dim], dtype=tf.float32, name="initial_states") #problem here
            inputs = tf.placeholder(shape=[footprint, 1, 21], dtype=tf.float32, name="input_data")


        def step(last_state, X):
            with tf.name_scope("To_insertion_and_IF_gates"):
                H_last = last_state
                process_rail = tf.concat([X, H_last], axis = 1, name = "process_rail_concat") #this makes the process rail
                insertion_gate = tf.add(tf.matmul(process_rail, W_Insertion, name = "matmul_insert"), B_Insertion, name = "bias_add_insert") #this makes the gate state
                input_forget_gate = tf.add(tf.matmul(process_rail, W_Input_Forget, name = "matmul_IF"), B_Input_Forget, name = "bias_add_IF") #this makes the other gate state
            with tf.name_scope("To_suggestion_gate"):
                insertion_gate = tf.sigmoid(insertion_gate, name = "sigmoid_insertion_gate") #this makes non-linear stuff
                input_forget_gate = tf.sigmoid(input_forget_gate, name = "sigmoid_IF")
                IF_negated = tf.scalar_mul(-1,input_forget_gate)  # this has to be here because it is after the nonlin
                input_gate = tf.add(tf.ones([1, hidden_dim]), IF_negated, name="making_input_gate") #making the two other gate states
                forget_gate = input_forget_gate #just to make sytax correct
            with tf.name_scope("To_suggestion_rail_and_node"):
                suggestion_rail_h = tf.multiply(insertion_gate, H_last, name = "suggestion_rail_making") #this one uses the insertion
                suggestion_rail = tf.concat([X, suggestion_rail_h], axis = 1, name = "making_suggestion_rail") #this one concatenates
                suggestion_gate = tf.add(tf.matmul(suggestion_rail, W_Suggestion, name = "matmul_suggestion"), B_Suggestion, name = "bias_add_suggestion")
                suggestion_gate = tf.tanh(suggestion_gate, name = "tanh_suggestion")#makes the suggestion gate
            with tf.name_scope("To_end"):
                suggestion = tf.multiply(suggestion_gate, input_gate, name = "to_suggestion_node") #this makes the filtered suggestion
                new_hidd = tf.multiply(forget_gate, H_last, name = "forget_gating") #but first, we need to forget some things
                new_hidd = tf.add(new_hidd, suggestion, name = "suggestion_adding")  #now, we add to form our next state

            return new_hidd

        with tf.name_scope("forward_roll"): #DO NOT CHANGE THIS NAME!!!
            #init_state = np.stack([init_state_cell, init_state_hidden])
            states_list = tf.scan(fn=step, elems=inputs, initializer=init_state, name="scan")
            curr_state = states_list[-1]
            pass_back_state = tf.add([0.0], states_list[0], name="pass_back_state")

        with tf.name_scope("prediction"):#do NOT CHANGE THIS NAME!!!
            current_hidden = curr_state
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

        next_state = np.zeros(shape=[1, hidden_dim])

        for epoch in range(epochs):
            reset, data = sm.next_epoch_waterfall()  # this gets you the entire cow, so to speak
            label = sm.get_label()
            label = np.reshape(label, [1, 1])
            data = np.reshape(data, [footprint, 1, 21])
            loss_ = 0

            if reset:  # this allows for hidden states to reset after the training set loops back around
                next_state = np.zeros(shape=[1,hidden_dim])

            next_state, loss_, _ = sess.run([curr_state, loss, optimizer],
                                            feed_dict={inputs: data, Y: label, init_state: next_state})

        RMS_loss = 0.0
        next_state = np.zeros(shape=[1, hidden_dim])
        # print(np.shape(next_state))
        for test in range(test_size):  # this will be replaced later

            data = sm.next_epoch_test_waterfall()
            label_ = sm.get_label()
            label = np.reshape(label_, [1, 1])
            data = np.reshape(data, [footprint, 1, 21])

            next_state, output_, loss_ = sess.run([pass_back_state, output, loss],
                                                  # why passback? Because we only shift by one!
                                                  feed_dict={inputs: data, Y: label, init_state: next_state})
            RMS_loss += np.sqrt(loss_)
            # carrier = [label_, output_[0][0], np.sqrt(loss_)]
            # test_logger.writerow(carrier)
        RMS_loss = RMS_loss / test_size
        print("test for " + str(SERIAL_NUMBER) + ": rms loss is ", RMS_loss)
        return RMS_loss