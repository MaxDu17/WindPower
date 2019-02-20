import tensorflow as tf
from pipeline.dataset_maker import SetMaker
import numpy as np
#this is for a RNN base test
class LSTM: #this isn't really an LSTM, but for the sake of polymorphism, it is...
    def __init__(self):
       print("LSTM object created")

    def graph(self, hyperparameters, sess):

        FOOTPRINT = hyperparameters[0]
        LEARNING_RATE = hyperparameters[1]
        cell_dim = hidden_dim = hyperparameters[2]
        epochs = hyperparameters[3]  # just a data issue. No data is being destroyed here. I'm just changing it to a compatible type
        test_size = hyperparameters[4]
        SERIAL_NUMBER = hyperparameters[5] # this is for telling which instance this is

        sm = SetMaker(FOOTPRINT)

        with tf.name_scope("weights_and_biases"):
            W_Hidden = tf.Variable(tf.random_normal(shape = [hidden_dim, hidden_dim]), name = "Hidden_weight")
            W_In = tf.Variable(tf.random_normal(shape = [1, hidden_dim]), name = "Input_weight")
            W_Out = tf.Variable(tf.random_normal(shape = [hidden_dim, 1], name = "Output_weight"))
            B_Hidden = tf.Variable(tf.zeros(shape = [1, hidden_dim]), name = "Hidden_bias")
            B_Out = tf.Variable(tf.zeros(shape = [1,1], name = "Output_bias"))

        with tf.name_scope("placeholders"):
            Y = tf.placeholder(shape=[1, 1], dtype=tf.float32, name="label")  # not used until the last cycle
            init_state = tf.placeholder(shape=[1, hidden_dim], dtype=tf.float32, name="initial_states")
            inputs = tf.placeholder(shape=[FOOTPRINT, 1 , 1], dtype=tf.float32, name="input_data") # we have 1,1 for matmul purposes

        def step(last_state, X):
            with tf.name_scope("propagation"):

                # output gate is not here, as it requires the changed cell state, which is not here yet
                H_last = last_state

                hidden_layer = tf.matmul(X, W_In, name = "Current_propagation")
                last_state_addition = tf.matmul(H_last, W_Hidden, name = "Past_propagation")
                hidden_layer = tf.add(hidden_layer, last_state_addition, name = "Combination")
                hidden_layer = tf.add(hidden_layer, B_Hidden, name = "Hidden_Bias_addition")
            return hidden_layer

        with tf.name_scope("forward_roll"):
            states_list = tf.scan(fn=step, elems=inputs, initializer=init_state,
                                  name="scan")  # funs step until inputs are gone
            curr_state = states_list[-1]  # used for the next time step
            pass_back_state = tf.add([0.0], states_list[0],
                                     name="pass_back_state")  # this is just making it a compute for tensroboard vis

        with tf.name_scope("prediction"):
            output = tf.matmul(curr_state, W_Out, name = "Output_propagation")
            output = tf.add(output, B_Out, name = "Output_Bias_addition")
            output = tf.nn.relu(output, name="output")

        with tf.name_scope("loss"):
            loss = tf.square(tf.subtract(output, Y))
            loss = tf.reduce_sum(loss)

        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

        sess.run(tf.global_variables_initializer())
        sm.create_training_set()
        summary = None  # this is just because it was used before

        next_state = np.zeros(shape=[1, cell_dim])

        for epoch in range(epochs):
            reset, data = sm.next_epoch_waterfall()  # this gets you the entire cow, so to speak
            label = sm.get_label()
            label = np.reshape(label, [1, 1])
            data = np.reshape(data, [FOOTPRINT, 1,1])
            loss_ = 0

            if reset:  # this allows for hidden states to reset after the training set loops back around
                next_state = np.zeros(shape=[1,cell_dim])

            next_state, loss_, _ = sess.run([curr_state, loss, optimizer],
                                            feed_dict={inputs: data, Y: label, init_state: next_state})

        RMS_loss = 0.0
        next_state = np.zeros(shape=[1, cell_dim])
        # print(np.shape(next_state))
        sm.reset_test_counter()
        for test in range(test_size):  # this will be replaced later

            data = sm.next_epoch_test_waterfall()
            label_ = sm.get_label()
            label = np.reshape(label_, [1, 1])
            data = np.reshape(data, [FOOTPRINT, 1,1])

            next_state, output_, loss_ = sess.run([pass_back_state, output, loss],
                                                  # why passback? Because we only shift by one!
                                                  feed_dict={inputs: data, Y: label, init_state: next_state})
            RMS_loss += np.sqrt(loss_)
            # carrier = [label_, output_[0][0], np.sqrt(loss_)]
            # test_logger.writerow(carrier)
        RMS_loss = RMS_loss / test_size
        print("test for " + str(SERIAL_NUMBER) + ": rms loss is ", RMS_loss)
        return RMS_loss
