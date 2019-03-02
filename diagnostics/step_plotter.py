import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
upper_bound = 500
lower_bound = 400
step_length = 1

FILE_NAME = "RNN_class"
x = np.arange(0,step_length*(upper_bound-lower_bound),step_length)

file_name_2 = "../Graphs_and_Results/" + FILE_NAME + "/GRAPHS/EVALUATE_TEST.csv"
#file_name_2 = "../Graphs_and_Results/" + FILE_NAME + "/data.csv"
data2 = pd.read_csv(file_name_2)

true_value = data2[["true_values"]]
predicted_values2 = data2[["predicted_values"]]

true = [k[0] for k in true_value.values]

predict2 = [n[0] for n in predicted_values2.values]
predict2_naive = np.roll(true, 1)


plt.step(x, true[lower_bound:upper_bound], label='truth')

plt.step(x, predict2[lower_bound:upper_bound], label=('predict version: ' + FILE_NAME))

title = FILE_NAME + " and truth" + ". " +\
        str(lower_bound)  + " to " + str(upper_bound) + ", step length " + str(step_length)

plt.title(title)

plt.legend()
plt.grid()
plt.show()