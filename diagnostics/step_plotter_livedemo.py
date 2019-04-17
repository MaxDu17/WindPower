import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
upper_bound = 500
lower_bound = 400
step_length = 1


def plot(truth, predicted):
        axes = plt.gca()
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 10)
        plt.bar(0.1, truth, width=0.2, color="0.5")
        plt.bar(0.5, predicted, width=0.2, color="0.1")
        plt.pause(0.0001)
        plt.cla()


FILE_NAME = "lstm_v2_c_class_FORE_AUTO"
LABEL_NAME = "lstm v2 Forecast and Compression"
x = np.arange(0,step_length*(upper_bound-lower_bound),step_length)

file_name_2 = "../Graphs_and_Results/" + FILE_NAME + "/GRAPHS/EVALUATE_TEST__.csv"
#file_name_2 = "../Graphs_and_Results/" + FILE_NAME + "/data.csv"
data2 = pd.read_csv(file_name_2)

true_value = data2[["true_values"]]
predicted_values2 = data2[["predicted_values"]]

true = [k[0] for k in true_value.values]

predict2 = [n[0] for n in predicted_values2.values]
predict2_naive = np.roll(true, 1)

plt.figure(num="graph")
plt.ion()

axes = plt.gca()
axes.set_xlim(0, 5)
axes.set_ylim(0, 10)

for i in range(999):
        plot(true[i], predict2[i])
        time.sleep(0.25)

'''
plt.step(x, true[lower_bound:upper_bound], label='truth')

plt.step(x, predict2[lower_bound:upper_bound], label=('predict version: ' + LABEL_NAME))

title = LABEL_NAME + " vs. Truth" + ". " +\
        str(lower_bound)  + " to " + str(upper_bound) + ", step length " + str(step_length)


plt.title(title)

plt.legend()
plt.grid()
plt.show()
'''