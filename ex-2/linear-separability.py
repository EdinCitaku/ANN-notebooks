import numpy as np
import numpy.random as rnd
import csv

def calculate_output(threshold, weight_matrix, x, my):
    inner_sum = 0
    print(weight_matrix)
    for i in range(len(x[0])):
        inner_sum+= weight_matrix[i]*x[my][i]
    return np.tanh(0.5*(inner_sum-threshold))

def calculate_energy_function(t,O):
    inner_sum = 0
    for my in len(t):
        inner_sum+= np.power(t[my] - O[my],2)
    return 0.5*inner_sum


def calculate_gradient(t, weight_matrix, x, threshold, learning_rate = 0.02, my_list = range(16)):

    #Let's do it without matrix multiplication first and later think of a solution where we dont have 3 for loops
    dw = np.zeros(w_ij.shape)
    for i in range(len(w_ij)):
        inner_sum = 0
        for my in my_list:
            O_my = calculate_output(threshold, weight_matrix,x, my)
            #Hier auf die Reihenfolge von x[i][my] vs x[my][i] achten!!!!
            inner_sum += (t[my] - O_my)*x[my][i]
        dw[i] = learning_rate*inner_sum
    return dw    

def esign(i):
    if i == 0:
        return 1
    return np.sign(i)
#Targe vectors
A =  [-1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1] 
B = [1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1] 
C = [-1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1] 
D = [1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1] 
E = [-1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1] 
F = [-1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1] 
G = [1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1]
#x vector
with open("input_data_numeric.csv","r") as f:
    x = list(csv.reader(f, delimiter=","))
    for idx, el in enumerate(x):
        x[idx] = list(map(int,el))
    x = np.array(x)[:,1:]
#Initialize variables with random inputs

#Lets test with A
for t in (A, B,C,D,E,F):
    for j in range(10):
        w_ij = (rnd.rand(4,)*0.4)-np.ones(4,)*0.2
        threshold = rnd.random_sample()*2 -1
        learning_rate = 0.02
        for i in range(100000):
            dw = calculate_gradient(t,w_ij,x, threshold)
            w_ij += dw
            #We need to test!
            finished = True
            for el in range(16):
                finished = finished and (t[el] == esign(calculate_output(threshold, w_ij, x, el)))
            if finished:
                break
        if finished:
            break
    print(finished)