import subprocess
import random
POPULATION_SIZE = 10
TRAINING_EPOCHS = 20000
TEST_SIZE = 1000


genetic_matrix = []

for i in range(POPULATION_SIZE):
    learning_rate = round(random.randrange(1, 20) * 0.0005, 6)
    footprint = int(random.randint(1, 15))
    cell_dim = hidden_dim = random.randint(1, 100)
    #hidden_dim = random.randint(1, 100) THIS IS FOR LATER
    genetic_matrix.append([footprint, learning_rate, cell_dim, hidden_dim, TRAINING_EPOCHS])
    subprocess.Popen(['/usr/bin/python3', './test.py', str(footprint), str(learning_rate), str(cell_dim), str(hidden_dim), str(TRAINING_EPOCHS), str(TEST_SIZE), str(i)])

print(genetic_matrix)
print("-----------------------------------")