import numpy as np
import pandas as pd
import csv

def load_csv_file():
    data = pd.read_csv("../Training_Sets/ALL_DATA_NORMALIZED.csv")
    return data

def main():
    dataframe = load_csv_file()
    dataframe.drop(["Minutes"], axis = 1, inplace=True)
    labels = dataframe[['power (MW)']].values.tolist()


    features = dataframe.values
    features_mat= np.asarray(features) #makes a features matrix
    covariance = np.cov(features_mat.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    compressed_eigen = eigenvectors[:,0]

    eigen_space_list = np.matmul(compressed_eigen.T,features_mat.T)
    print(eigenvalues)
    print(eigenvalues/sum(eigenvalues))
    final_PCA = eigen_space_list.T
    input()
    k = open("../Training_Sets/ALL_DATA_PCA.csv", "w")
    final = csv.writer(k, lineterminator="\n")
    final_list = final_PCA.tolist()
    final.writerow(['PCA', 'Power'])
    for i in range(len(labels)):
        final.writerow([final_list[i], labels[i][0]])


if __name__ == '__main__':
    main()
    print("I'm done")