from pipeline.data_feeder_forecast import DataParser_Forecast
from pipeline.dataset_maker_forecast import SetMaker_Forecast
dp = DataParser_Forecast()
print(dp.grab_list_range(0,10))

SM = SetMaker_Forecast(2)

SM.create_training_set()

print("--------------------------------------")
print("This tests the validation set generator")
SM.create_validation_set()
for i in range(10):
    print(SM.next_epoch_valid_waterfall())
    print(SM.get_label())
    print("\n")

print("--------------------------------------")
print("This tests the training set generator")
for i in range(10):
    print(SM.next_epoch_waterfall())
    print(SM.get_label())
    print("\n")

print("--------------------------------------")
print("This tests the test set generator")
for i in range(10):
    print(SM.next_epoch_test_waterfall())
    print(SM.get_label())
    print("\n")
