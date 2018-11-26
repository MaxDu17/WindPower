import subprocess
import random
population_size = 10
training_epochs = 20000


genetic_matrix = []

for i in range(population_size):
    learning_rate = round(random.randrange(1, 20) * 0.0005, 6)
    footprint = int(random.randint(1, 15))
    cell_dim = hidden_dim = random.randint(1, 100)
    #hidden_dim = random.randint(1, 100) THIS IS FOR LATER
    genetic_matrix.append([footprint, learning_rate, cell_dim, hidden_dim, training_epochs])
    subprocess.Popen(['/usr/bin/python3', './test.py', str(footprint), str(learning_rate), str(cell_dim), str(hidden_dim), str(training_epochs)])

print(genetic_matrix)
print("-----------------------------------")