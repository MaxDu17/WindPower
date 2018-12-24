import subprocess
import random
import csv
POPULATION_SIZE = 7
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
    genetic_matrix.append([footprint, learning_rate, cell_dim, hidden_dim, TRAINING_EPOCHS, TEST_SIZE, i])

    subprocess_array.append(
        subprocess.Popen(['/usr/bin/python3', '../Models/lstm_v2_c_genetic.py', str(footprint),
                          str(learning_rate), str(cell_dim), str(hidden_dim), str(TRAINING_EPOCHS), str(TEST_SIZE), str(i)]))

print(genetic_matrix)
exit_codes=[p.wait() for p in subprocess_array]

for i in range(POPULATION_SIZE):
    data = open(str(i)+".csv", "r")
    data_ = list(csv.reader(data, lineterminator = "\n"))
    data_ = [m[0] for m in k]
    print(data_)
    loss = float(data_[0])
    data_dict[i] = [genetic_matrix, loss]
    data.close()

test = open("test.csv", "w")
test_ = csv.writer(test, lineterminator= "\n")
[test_.writerows(k) for k in data_dict]
print("DONE DONE DONE DONE DONE DONE")

