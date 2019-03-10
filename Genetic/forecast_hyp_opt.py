'''
This code will genetically optimize a class-defined LSTM model
because of the class structure, there should be no problem with any network
'''
import random
import csv
import tensorflow as tf


############# CHANGE ME ###########
from Models.lstm_v9_c_class_FORE import LSTM
name = "lstm_v9_c_class_FORE"
###################################


POPULATION_SIZE = 10
TRAINING_EPOCHS = 3000 #used to be 500
TEST_SIZE = 200
ACTIVE_HYP = 3
CROSSOVER = 3
GENETIC_EPOCHS = 20
MUTATION_RATE = 0.3


genetic_matrix = []
data_dict = {}
subprocess_array = []

model = LSTM()#this makes the entire model for genetic only


def sort_second(val): #sorting function
    return val[1]

def is_mutate(): #mutuation oracle
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

        if value == 0: #this is a "dumb" way to prevent prevent divergence to zero
            value += 1

    return value  # returns the modified value

def mutate_float(value):
    mutation = is_mutate()
    if mutation:
        random_shift = random.uniform(-0.002, 0.002)
        #print(random_shift)
        if(random_shift + value > 0):
            value += random_shift
        else:
            print("exceeds bounds, skipping mutation")

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
                cell_hidden_dim = random.randint(10, 150)
                genetic_matrix = [footprint, learning_rate, cell_hidden_dim, TRAINING_EPOCHS, TEST_SIZE, i]
                results.append([genetic_matrix, model.graph(hyperparameters = genetic_matrix, sess = sess)])

            else:
                children[i].append(TRAINING_EPOCHS)
                children[i].append(TEST_SIZE)
                children[i].append(i)
                results.append([children[i], model.graph(children[i], sess)])


        results.sort(key = sort_second)
        results = [k[0] for k in results] #removes the error. This is no longer needed
        results = [k[0:3] for k in results] #removes the serial number. This is no longer needed
        children = cross_over(results[0], results[1]) #this should g et the hyperparameters
        first = False

        print(children)
        print("The kept parents are: " + str(results[0:2]))

    k = open(name + "best.csv", "w")
    best_writer = csv.writer(k, lineterminator = "\n")
    best_writer.writerows(results[0:2])


