import csv
k_ = open("MARKEDTOTALSET.csv", 'r')
k = list(csv.reader(k_))
k = [m[0] for m in k]

best_run = 0
best_run_upper = 0
best_run_lower = 0
current_run = 0
for i in range(len(k)):
    if k[i] != "DATA NOT FOUND":
        current_run += 1
    else:
        if current_run > best_run:
            best_run = current_run
            best_run_lower = (i-best_run)
            best_run_upper = i-1
            current_run = 0
        else:
            current_run = 0

print("here is the best run length: " + str(best_run))
print("here is the index range of this best run: [" + str(best_run_lower) + "," + str(best_run_upper) + "]")
print("in excel, that is " "[" + str(best_run_lower+2) + "," + str(best_run_upper+2) + "]")