import numpy as np
from numpy import random as rnd
import csv
import time

def calculate_output(threshold, weight_matrix, x):
    return np.tanh(np.dot(weight_matrix,x)-threshold)

def create_layer(inputsize, layersize):
    neurons = np.zeros((layersize,1))
    neurons = neurons.reshape(layersize,1)
    w_ij = rnd.normal(size=(layersize, inputsize))
    return (neurons, w_ij)

def calculate_classification_error(V,w,O, validation_set):
    inner_sum = 0
    pval = len(validation_set)
    for my in range(pval):
        V[0] = np.array(validation_set[my])[:2]
        V[0] = V[0].reshape(2,1)
        t_my = validation_set[my][2]
        #Propagate Forward
        for l in range(1,4):
            V[l]= calculate_output(O[l-1],w[l-1],V[l-1])
        inner_sum += np.abs(esign(V[3])-t_my)
    return inner_sum*0.5*(1/pval)

def dg_neuron(x,w,O):
    ret = (1-np.power(np.tanh(np.dot(w,x) - O),2))
    return ret

def dg(x,w,O):
    return (1-np.power(np.tanh(np.dot(w,x) - O),2))

def calculate_b(w, x, j, O):
    inner_sum = 0
    for k in range(len(x)):
        inner_sum+= w[j][k]*x[k] - O[j]
    return inner_sum



def esign(i):
    if i == 0:
        return 1
    return np.sign(i)

#This is the input from our function
firstlayersize = 2 
#The size of the next 2 values we need to experiment with
M1 = 20
M2 = 10
#This is the output of our function
outputsize = 1

#The threshold, we just leave them at 0 for the beginning
O1 = np.zeros((M1,1))
O2 = np.zeros((M2,1))
O3 = np.zeros((outputsize,1))

inputlayer = np.zeros((firstlayersize,1))
layer1, weight_matrix1 = create_layer(firstlayersize, M1)
layer2, weight_matrix2 = create_layer(M1, M2)
outputlayer, weight_matrix3= create_layer(M2, outputsize)

V = [inputlayer, layer1,layer2, outputlayer]
w = [weight_matrix1,weight_matrix2, weight_matrix3]
O = [O1,O2,O3]
#Import training and validation set
#TODO Look at how these sets look like and fix the formatting if needed 
with open("/home/edin/uni/ANN/ex-2/training_set.csv","r") as f:
    training_set = list(csv.reader(f, delimiter=","))
    for idx, el in enumerate(training_set):
        training_set[idx] = list(map(float,el))
    training_set = np.array(training_set)

with open("/home/edin/uni/ANN/ex-2/validation_set.csv","r") as f:
    validation_set = list(csv.reader(f, delimiter=","))
    for idx, el in enumerate(validation_set):
        validation_set[idx] = list(map(float,el))
    validation_set = np.array(validation_set)

#TODO implement learning algorithm
#I chose T randomly in this case
T = 100*len(training_set)
C = 1
start = time.time()
learning_rate = 0.02
for el in range(T):
    my = rnd.choice(range(len(training_set)))
    V[0] = np.array(training_set[my])[:2]
    V[0] = V[0].reshape(2,1)
    t = np.array(training_set[my])[2]
    #Propagate Forward
    for l in range(1,4):
        V[l]= calculate_output(O[l-1],w[l-1],V[l-1])
    #Calculate Error
    #Initiate d
    #Propagate Backwards
    delta_O = dg_neuron(V[2], w[2], O[2]) * (t - V[3])
    delta_V2 = dg(V[1], w[1], O[1])
    delta_V2 = delta_O* w[2].T*delta_V2
    print(w[2])
    delta_V1 = dg(V[0], w[0], O[0])
    delta_V1 = (w[1].T*delta_V1)

    delta_V1 = np.dot(delta_V1,delta_V2)
    d = [delta_V1,delta_V2,delta_O]
    for l in range(3):
        w[l] +=  learning_rate*d[l]*V[l].T
        O[l] -= learning_rate*d[l]
    if el % 10000 == 0:
        C = calculate_classification_error(V,w,O,validation_set)[0]
        print(C)
    if C < 0.12:
        break
        
end = time.time()
print(end - start)
#Lets save everything
for i in range(3):
    np.savetxt(f"w{i+1}.csv",w[i],delimiter=',')
    np.savetxt(f"t{i+1}.csv",O[i],delimiter=',')
print("Finished!")
