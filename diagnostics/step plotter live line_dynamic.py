import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import time
upper_bound = 10
lower_bound = 0
step_length = 1
fig, ax = plt.subplots()

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
plt.ylim(-1, 10)

line, = ax.step(x, true[0:10])
line2, = ax.step(x, predict2[0:10])
ax.set_title("Compression w/Forecast Step_Dynamic")
plt.ylabel("Ignore these axes")

def animate(i):
    k = max(true[i:i+10])
    l = max(predict2[i:i + 10])

    if(k>l):
        plt.ylim(-1, k+1)
    else:
        plt.ylim(-1, l+1)

    a = str(np.round(true[i], 3))

    while len(a) < 4:
        a = a + "0"
    b = str(np.round(predict2[i],3))

    while len(b) < 4:
        b = b + "0"

    print(a + "\t\t" + b)
    line.set_ydata(true[i:i+10])  # update the data
    line2.set_ydata(predict2[i:i + 10])  # update the data
    return line, line2,


# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    line2.set_ydata(np.ma.array(x, mask=True))
    return line, line2

ani = animation.FuncAnimation(fig, animate, init_func=init,
                              interval=120, blit=True)
plt.show()