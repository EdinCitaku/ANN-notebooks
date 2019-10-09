import numpy as np
from numpy import random as rnd
import csv

def calculate_output(threshold, weight_matrix, x, j):
    inner_sum = 0

    for i in range(len(x)):
        inner_sum+= weight_matrix[j][i]*x[i]
    return np.tanh(0.5*(inner_sum-threshold[j]))

def create_layer(inputsize, layersize):
    neurons = np.zeros((layersize,))
    w_ij = rnd.normal(size=(layersize, inputsize))
    return (neurons, w_ij)

def calculate_classification_error(V,w,O, validation_set):
    inner_sum = 0
    pval = len(validation_set)
    for my in range(pval):
        V[0] = np.array(validation_set[my])[:2]
        t_my = validation_set[my][2]
            #Propagate Forward
        for l in range(1,4):
            for k in range(len(V[l])):
                V[l][k] = calculate_output(O[l-1],w[l-1],V[l-1],k)
        inner_sum += np.abs(esign(V[3])-t_my)
    return inner_sum*0.5*(1/pval)

def dg(b):
    return (1-np.power(np.tanh(b),2))

def calculate_b(w, x, j, O):
    inner_sum = 0
    for k in range(len(x)):
        inner_sum+= w[j][k]*x[k] - O
    return inner_sum



def esign(i):
    if i == 0:
        return 1
    return np.sign(i)

#This is the input from our function
firstlayersize = 2 
#The size of the next 2 values we need to experiment with
M1 = 16
M2 = 7
#This is the output of our function
outputsize = 1

#The threshold, we just leave them at 0 for the beginning
O1 = np.zeros((M1,))
O2 = np.zeros((M2,))
O3 = np.zeros((outputsize,))

inputlayer = np.zeros((firstlayersize,))
layer1, weight_matrix1 = create_layer(firstlayersize, M1)
layer2, weight_matrix2 = create_layer(M1, M2)
outputlayer, weight_matrix3= create_layer(M2, outputsize)

V = [inputlayer, layer1,layer2, outputlayer]
w = [weight_matrix1,weight_matrix2, weight_matrix3]
O = [O1,O2,O3]
#Import training and validation set
#TODO Look at how these sets look like and fix the formatting if needed 
with open("training_set.csv","r") as f:
    training_set = list(csv.reader(f, delimiter=","))
    for idx, el in enumerate(training_set):
        training_set[idx] = list(map(float,el))
    training_set = np.array(training_set)

with open("validation_set.csv","r") as f:
    validation_set = list(csv.reader(f, delimiter=","))
    for idx, el in enumerate(validation_set):
        validation_set[idx] = list(map(float,el))
    validation_set = np.array(validation_set)

#TODO implement learning algorithm
#I chose T randomly in this case
T = 1000
print(len(training_set))
learning_rate = 0.02
for el in range(T):
    my = rnd.choice(range(len(training_set)))
    V[0] = np.array(training_set[my])[:2]
    #Propagate Forward
    for l in range(1,4):
        for k in range(len(V[l])):
            
            V[l][k] = calculate_output(O[l-1],w[l-1],V[l-1],k)
    #Calculate Error
    #Initiate d
    d = []
    for i in range(3):
        d.append(np.zeros((len(w[i]),)))
    b_L = calculate_b([w[2]],V[3],0,O[2])
    t = training_set[my][2]
    d_L = dg(b_L)*(t-V[3])
    d.append([d_L])    
    #print(f"V[0]: {V[0]}, V[3]: {V[3]}, t :{t} ")
    #Propagate Backwards
    for l in range(4,2):
        for j in range(len(d[l-1])):
            b = calculate_b(w[l],V[l],j,O)
            dgb = dg(b)
            inner_sum = 0
            for i in range(len(d[l])):
                inner_sum = d[l][i]*w[l][i][j]*dgb
            d[l-1][j] = inner_sum
    #Update weights and threshold
    for l in range(3):
        w[l] +=  learning_rate*d[l][:,None]*V[l]
        O[l] -= learning_rate*d[l]
#Lets calculate our error probability
C = calculate_classification_error(V,w,O,training_set)

print(C)