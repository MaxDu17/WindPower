import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
upper_bound = 500
lower_bound = 0
step_length = 1
version_number_1 = 1

x = np.arange(0,step_length*(upper_bound-lower_bound),step_length)

#file_name_1 = "2012/v" + str(version_number_1) + "/GRAPHS/v14/TEST.csv"
#file_name_1 = "2012/v" + str(version_number_1) + "/GRAPHS/EVALUATE_TEST.csv"
file_name_1 = "LSTM_code/control/test.csv"
data1 = pd.read_csv(file_name_1)

true_value = data1[["true_values"]]
predicted_values1 = data1[["predicted_values"]]


true = [k[0] for k in true_value.values]
#print(true)
predict1 = [n[0] for n in predicted_values1.values]
#print(predict1)


plt.step(x, predict1[lower_bound:upper_bound], label=('predict version ' + str(version_number_1)))
plt.step(x, true[lower_bound:upper_bound], label='truth')

title = "version " + str(version_number_1) + "-"  +\
        str(lower_bound)  + " to " + str(upper_bound) + ", step length " + str(step_length)

plt.title(title)

plt.legend()
plt.grid()
plt.show()



