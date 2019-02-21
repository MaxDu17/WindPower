import tensorflow as tf
from pipeline.dataset_maker_forecast import SetMaker_Forecast
import numpy as np

class LSTM:
    def __init__(self):
       print("LSTM created")

    def graph(self, hyperparameters, sess):

        footprint = hyperparameters[0]
        learning_rate = hyperparameters[1]
        cell_dim = hidden_dim = hyperparameters[2]
        #hidden_dim = hyperparameters[3]
        epochs = hyperparameters[3]  # just a data issue. No data is being destroyed here. I'm just changing it to a compatible type
        test_size = hyperparameters[4]
        SERIAL_NUMBER = hyperparameters[5] # this is for telling which instance this is

        sm = SetMaker_Forecast(footprint)
        with tf.name_scope("weights_and_biases"):
            W_Forget = tf.Variable(tf.random_normal(shape=[hidden_dim + 21, cell_dim]), name="forget_weight")
            #W_Forget = tf.Variable(tf.random_normal(shape=[cell_dim, hidden_dim + 1]), name="forget_weight")
            W_Output = tf.Variable(tf.random_normal(shape=[hidden_dim + 21, cell_dim]), name="output_weight")
            W_Gate = tf.Variable(tf.random_normal(shape=[hidden_dim + 21, cell_dim]), name="gate_weight")
            W_Input = tf.Variable(tf.random_normal(shape=[hidden_dim + 21, cell_dim]), name="input_weight")
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
            inputs = tf.placeholder(shape=[footprint, 1, 21], dtype=tf.float32, name="input_data")


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

        next_state = np.zeros(shape=[2, 1, cell_dim])

        for epoch in range(epochs):
            reset, data = sm.next_epoch_waterfall()  # this gets you the entire cow, so to speak
            label = sm.get_label()
            label = np.reshape(label, [1, 1])
            data = np.reshape(data, [footprint, 1, 21])
            loss_ = 0

            if reset:  # this allows for hidden states to reset after the training set loops back around
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