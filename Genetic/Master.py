import subprocess
import random
import csv
POPULATION_SIZE = 10
TRAINING_EPOCHS = 100
TEST_SIZE = 1000


genetic_matrix = []
data_dict = {}
subprocess_array = []
for i in range(POPULATION_SIZE):
    learning_rate = round(random.randrange(1, 20) * 0.0005, 6)
    footprint = int(random.randint(1, 15))
    cell_dim = hidden_dim = random.randint(1, 100)
    #hidden_dim = random.randint(1, 100) THIS IS FOR LATER
    genetic_matrix.append([footprint, learning_rate, cell_dim, hidden_dim, TRAINING_EPOCHS, TEST_SIZE])

    k= subprocess.Popen(['/usr/bin/python3', '../Models/lstm_v2_c_genetic.py', str(footprint),
                          str(learning_rate), str(cell_dim), str(hidden_dim), str(TRAINING_EPOCHS), str(TEST_SIZE), str(i)])
    k.wait()

print(genetic_matrix)
#exit_list = [p.wait for p in subprocess_array]


for i in range(POPULATION_SIZE):
    data = open(str(i)+".csv", "r")
    data_ = list(csv.reader(data, lineterminator = "\n"))
    data_ = [m[0] for m in data_]
    loss = float(data_[0])
    data_dict[i] = [genetic_matrix[i], loss]
    data.close()

test = open("test.csv", "w")
test_ = csv.writer(test)
print(data_dict)
for k, v in data_dict.items():
    test_.writerow([k, v])
print("DONE DONE DONE DONE DONE DONE")

