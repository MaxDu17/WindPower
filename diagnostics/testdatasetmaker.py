from pipeline import SetMaker
import time
sm = SetMaker()

sm.create_training_set()
sm.next_epoch_test()

input("This program makes sure that the LSTM is being honest.\nIt checks the same function "
              "that is responsible for feeding the data during real testing, and emulates the testing program.\nPress enter to continue.\n")

print("\t\t\tTHE OUTPUT IS INTENTIONALLY SLOWED DOWN\n")
for i in range(10):
    time.sleep(1)
    data, label = sm.next_epoch_test_pair()
    print("epoch " + str(i))
    print("Data point fed in: " + str(data))
    print("Label/next data point: " + str(label))
    print("----------------------------------------------------------------\n")
