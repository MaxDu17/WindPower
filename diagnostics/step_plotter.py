import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
upper_bound = 500
lower_bound = 0
step_length = 1

FILE_NAME = "lstm_v2_c_CSV_FED"
x = np.arange(0,step_length*(upper_bound-lower_bound),step_length)

file_name_1 = "../Graphs_and_Results/Naive_case.csv"
file_name_2 = "../Graphs_and_Results/" + FILE_NAME + "/GRAPHS/EVALUATE_TEST.csv"
data2 = pd.read_csv(file_name_2)
data1 = pd.read_csv(file_name_1)

print(data2[["true_values"]])
print(data1[["true_values"]])
true_value = data1[["true_values"]]
predicted_values1 = data1[["predicted_values"]]
predicted_values2 = data2[["predicted_values"]]

true = [k[0] for k in true_value.values]

predict1 = [n[0] for n in predicted_values1.values]
predict2 = [n[0] for n in predicted_values2.values]


plt.step(x, true[lower_bound:upper_bound], label='truth')
#plt.step(x, predict1[lower_bound:upper_bound], label=('predict version ' + str(version_number_1)))
plt.step(x, predict2[lower_bound:upper_bound], label=('predict version: ' + FILE_NAME))

title = FILE_NAME + " and naive" + ". " +\
        str(lower_bound)  + " to " + str(upper_bound) + ", step length " + str(step_length)

plt.title(title)

plt.legend()
plt.grid()
plt.show()