from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
import csv
#from sklearn.metrics import mean_squared_error


def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')

def abs_error(test, prediction):
    big_error = 0
    for i in range(len(test)):
        big_error += abs(prediction[i] - test[i])
    big_error = big_error/len(test)
    return big_error


data = read_csv("../Training_Sets/ARIMA.csv", skiprows=3)  # read file
power_ds = data[["power (MW)"]]

X = power_ds.values
size = 2000 #this is meant to be a fair comparison
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
test_ = open('../Graphs_and_Results/ARIMA/data.csv', "w")
test_logger = csv.writer(test_, lineterminator="\n")
print(len(test))
test_logger.writerow(["predicted_values", "true_values"])
for t in range(len(test)):
    #model = ARIMA(history, order=(5, 1, 0))
    model = ARIMA(history, order=(1, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    test_logger.writerow([yhat[0], obs[0]])
    print('predicted=%f, real=%f' % (yhat, obs))
error = abs_error(test, predictions)
print('Test AE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()