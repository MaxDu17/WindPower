import subprocess
import random
population_size = 10
training_epoch = 20000


genetic_matrix = []

for i in range(population_size):
    footprint = int(random.randint(1, 15))
    learning_rate = round(random.randrange(1, 20)*0.0005, 6)
    cell_dim = random.randint(1, 100)
    hidden_dim = random.randint(1, 100)
    genetic_matrix.append([footprint, learning_rate, cell_dim, hidden_dim])
    subprocess.Popen(['/usr/bin/python3', './test.py', str(footprint), str(learning_rate), str(cell_dim), str(hidden_dim)])

print(genetic_matrix)
print("-----------------------------------")