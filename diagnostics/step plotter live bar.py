import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
upper_bound = 10
lower_bound = 0
step_length = 1
fig, ax = plt.subplots()

FILE_NAME = "lstm_v2_c_class_FORE_AUTO"
LABEL_NAME = "lstm v2 Forecast and Compression"
x_label = ('TRUE', 'PREDICT')
x = np.arange(0,step_length*(upper_bound-lower_bound),step_length)
index = np.arange(2)
file_name_2 = "../Graphs_and_Results/" + FILE_NAME + "/GRAPHS/EVALUATE_TEST__.csv"
#file_name_2 = "../Graphs_and_Results/" + FILE_NAME + "/data.csv"
data2 = pd.read_csv(file_name_2)

true_value = data2[["true_values"]]
predicted_values2 = data2[["predicted_values"]]

true = [k[0] for k in true_value.values]

predict2 = [n[0] for n in predicted_values2.values]
predict2_naive = np.roll(true, 1)
plt.ylim(-1, 10)

line, = ax.bar(x = 0,height = true[0], label = "Truth")
line2, = ax.bar(x = 1, height = predict2[0], label = "Prediction")
ax.axes.get_xaxis().set_visible(False)
text = ax.text(1, 9, "test", fontsize=12)
ax.set_title("Prediction and Truth on Wind Power")

def animate(i):

    line.set_height(true[i])  # update the data
    line2.set_height(predict2[i])  # update the data

    print(str(np.round(true[i], 3)) + "\t\t" +str(np.round(predict2[i],3)))

    return line, line2,


# Init only required for blitting to give a clean slate.
def init():
    line.set_height(np.ma.array(x, mask=True))
    line2.set_height(np.ma.array(x, mask=True))
    return line, line2

ani = animation.FuncAnimation(fig, animate, init_func=init,
                              interval=100, blit=True)
plt.show()