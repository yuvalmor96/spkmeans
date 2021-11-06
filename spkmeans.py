import pandas as pd
import numpy as np
import sys
import myspkmeans as mspk
from enum import Enum


class goals(Enum):
    spk = "spk"
    wam = "wam"
    ddg = "ddg"
    lnorm = "lnorm"
    jacobi = "jacobi"


def main():
    K = int (sys.argv[1])
    if (not (K>=0)):
        print("Invalid Input!")
        assert(K>=0)
    max_iter = 300
    goal = sys.argv[2]
    if (not (goal in goals._value2member_map_)):
        print("Invalid Input!")
        assert(goal in goals._value2member_map_)  
    file_name = sys.argv[3]
    vector_list = readfile(file_name)
    vector_list.index = vector_list.index.astype('int64')
    vector_num = vector_list.shape[0]
    vector_len = vector_list.shape[1]
    vector_list_to_C = vector_list.values.tolist()
    if (K==0):
        newK = mspk.heuristic(vector_list_to_C, vector_num,vector_len)
        K = newK

    if (goal=="spk"):
        new_vector_list = mspk.fullSpectralPy(K,vector_num,vector_list_to_C)
        new_vector_list = pd.DataFrame(new_vector_list)
        vector_num = new_vector_list.shape[0]
        centroids_for_C, indices = init_centroids(K, new_vector_list.values,new_vector_list.index,vector_num) #new_vector_list
        new_vector_list_to_C = new_vector_list.values.tolist()
        final_centroids = mspk.fit(centroids_for_C,new_vector_list_to_C,max_iter,K)
        output_centroids = mspk.kmeans(K, vector_num, vector_len, new_vector_list_to_C, final_centroids)
        converted_indexes = [str(elm) for elm in indices]
        joined_indexes = ",".join(converted_indexes)
        print(joined_indexes)
        output_centroids = mat_neg_zeros_eraser(output_centroids)
        printMat(output_centroids)
    elif (goal=="wam"):
        matrix = mspk.WeightedAdjacencyMatrix(vector_list_to_C, vector_num, vector_len)
        matrix = mat_neg_zeros_eraser(matrix)
        printMat(matrix)
    elif (goal=="ddg"):
        weighted_matrix = mspk.WeightedAdjacencyMatrix(vector_list_to_C, vector_num, vector_len)
        diagonal_matrix = mspk.DiagonalDegreeMatrix(weighted_matrix, vector_num, vector_len)
        diagonal_matrix = mat_neg_zeros_eraser(diagonal_matrix)
        printMat(diagonal_matrix)

    elif (goal=="lnorm"):
        matrix = mspk.NormalizedGraphLaplacian(vector_list_to_C, vector_num, vector_len)
        matrix = mat_neg_zeros_eraser(matrix)
        printMat(matrix)

    elif (goal=="jacobi"):
        eigenvalues, eigenvectors = mspk.Jacobi(vector_list_to_C, vector_num, vector_len)
        for i in range(len(eigenvalues)):      # erasing minus zeros for eigenvalues
            if ((eigenvalues[i]<0) and (eigenvalues[i]>-0.00005)):
                eigenvalues[i] = 0
        print(",".join(["%.4f" % float(i) for i in eigenvalues]))
        eigenvectors = mat_neg_zeros_eraser(eigenvectors)
        printMat(eigenvectors)



def printMat(matrix):
    for i in range(len(matrix)):
            if i==(len(matrix)-1):
                print(",".join(["%.4f" % float(i) for i in matrix[i]]),end="")
            else:
                print(",".join(["%.4f" % float(i) for i in matrix[i]]))



def init_centroids(k,vector_list,vector_list_ind,vector_num):
    np.random.seed(0)
    if (not (k<vector_num)):
        print("Invalid Input!")
        assert(k<vector_num)
    dist = [0 for i in range(vector_num)]
    centroid_for_C = [0 for i in range(k)]
    centroids_index = [0 for i in range(k)]
    #first centroid
    rand_index = np.random.choice(vector_num)
    centroid_for_C[0]= vector_list[rand_index]
    centroids_index[0] = vector_list_ind[rand_index]
    z=1
    while z<k:
        #calc dist
        for i in range(vector_num):
            if z==1:
                 dist[i] = distance(vector_list[i], centroid_for_C[0])
            else:
                dist[i] = min_dist_centroid(vector_list,i,centroid_for_C,z,dist)
        #calc prob
        sum = np.sum(dist)
        prob = dist/sum
        # find new centroid
        chosen_ind = np.random.choice(vector_num,p=prob)
        centroid_for_C[z]= vector_list[int(chosen_ind)]
        centroids_index[z] = vector_list_ind[int(chosen_ind)]
        z+=1
    #convert to python lists
    for i in range(k):
        centroid_for_C[i] = centroid_for_C[i].tolist()
    return centroid_for_C, centroids_index

def readfile(filename):
    file = pd.read_csv(filename,header=None)
    return file


def distance(v1,v2):
    return  np.sum((v1-v2)**2)

def min_dist_centroid(vector_list,i,centroid_for_C,z,dist):
    return min(distance(vector_list[i], centroid_for_C[z-1]), dist[i])

def mat_neg_zeros_eraser(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if ((matrix[i][j]>-0.00005) and (matrix[i][j]<0)):
                matrix[i][j] = 0
    return matrix

if(__name__ == "__main__"):
    main()