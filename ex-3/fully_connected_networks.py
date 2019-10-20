import numpy as np
from numpy import random as rnd
import csv

def unpickle(file):
    "Returns an a dictionary of a single batch, if filename is correct"
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def calculate_mean(x_train):
    "Calculate the mean of a set of training data"
    mean = np.mean(x_train,axis=0)
    return mean


def center_data(x_train):
    #TODO make sure that this works as intended
    mean = calculate_mean(x_train)
    mean = mean.astype(np.uint8)
    centered_data = x_train - mean
    centered_data = centered_data.astype(np.uint8)
    return centered_data

def sigmoid_activation_function(b):

    e = np.exp(-b)
    temp = 1+e
    temp = 1/temp
    return temp

def g(b):
    #return np.tanh(b)    inner_sum = inner_sum.reshape((len(inner_sum),1))
    return np.tanh(b)
    return sigmoid_activation_function(b)

def dg(b):
    return (1-np.power(np.tanh(b),2))

    o_b = sigmoid_activation_function(b)
    #o_b= (1-np.power(np.tanh(b),2))
    #return o_b
    return o_b*(1-o_b)

def create_layer(inputsize, layersize):
    neurons = np.zeros((layersize,1))
    neurons = neurons.reshape(layersize,1)
    w_ij = rnd.normal(size=(layersize, inputsize))
    return (neurons, w_ij)

def esign(i):
    if i == 0:
        return 1
    return np.sign(i)

def calculate_output(threshold, weight_matrix, x):
    inner_sum = np.dot(weight_matrix,x)
    inner_sum = inner_sum.reshape((len(inner_sum),x.shape[1]))
    inner_sum = inner_sum - threshold
    ret = g(inner_sum)
    assert(ret.shape == threshold.shape)
    return ret

def calculate_output_batch(threshold, weight_matrix, V, batchsize):
    
    
    inner_sum = np.dot(weight_matrix,V)
    #inner_sum = inner_sum.reshape((len(inner_sum),x.shape[1]))
    inner_sum = inner_sum.reshape((threshold.shape[0],batchsize))
    inner_sum = inner_sum - threshold
    ret = g(inner_sum)
    assert(ret.shape == (threshold.shape[0],batchsize))
    return ret



def propagate_forward(V : list, O : list, w : list, batchsize : int) -> None:

    #TODO make sure that it works as intended

    assert(len(V)-1==len(O) and len(O)==len(w))

    for l in range(1,len(V)):
        V[l]= calculate_output_batch(O[l-1],w[l-1],V[l-1], batchsize)


def calculate_b(w, x, O):
    inner_sum = np.dot(w,x)
    inner_sum = inner_sum - O
    return inner_sum

def calculate_d_L(b, t, V):
    
    #TODO make sure it works properly
    return dg(b)(t-V)

def calculate_single_d(d_l, w, b):
    
    #TODO make sure that the order of multiplications is right
    dgb = dg(b)
    inner_product = np.dot(d_l.T, w).T
    ret = np.multiply(inner_product, dgb)
    return ret


def calculate_all_d(V, O, w, t):
    
    #TODO make sure that it works as intended
    L = len(O)-1
    b_L = calculate_b(w[L],V[L], O[L])
    b_L = dg(b_L)
    temp = (t-V[-1])
    delta_start = b_L*temp
    d = [[]]*(L+1)
    d[L] = delta_start
    for i in range(L, 0, -1):
        #TODO make sure the range function works as intended
        b =  calculate_b(w[i-1],V[i-1],O[i-1])
        delta = calculate_single_d(d[i],w[i],b)
        d[i-1] = delta
    return d



def propagate_backwards(V : list, O : list, w : list, t : int, learning_rate: float, batchsize : int):
    #TODO include support for batchsize
    L = len(w)
    d = calculate_all_d(V,O,w,t) 
    for l in range(L):
        d_w = learning_rate*(d[l]@V[l].T)
        d_O = learning_rate* (d[l]@np.ones((batchsize,1)))
        w[l] +=  d_w
        O[l] -= d_O

def create_multible_layers(*argv):
    w = []
    V = []
    O = []
    inputlayer = np.zeros((argv[0],1))
    V.append(inputlayer)
    old_layersize = argv[0]
    for layersize in argv[1:]:
        O_i = np.zeros((layersize,1))
        V_i, w_i = create_layer(old_layersize,layersize)
        old_layersize = layersize    
        w.append(w_i)
        V.append(V_i)
        O.append(O_i)
    return (w,V,O)
"""
OLD VERSION OF THE FUNCTION
def calculate_classification_error(V,w,O, validation_set):
    inner_sum = 0
    pval = len(validation_set)
    for my in range(pval):
        V[0] = np.array(validation_set[my])[:2]
        V[0] = V[0].reshape(2,1)
        t_my = validation_set[my][2]
        propagate_forward(V,O,w)
        inner_sum += np.abs(esign(V[3])-t_my)
    return inner_sum*0.5*(1/pval)
"""


def learn(learning_rate:float, data: list, labels: list,test_data : list, test_labels : list, batchsize: int, *argv):

    w,V,O = create_multible_layers(*argv)
    #merged = np.concatenate([data, labels], axis=1)
    for T in range(1000):
        #Permutate the data in the beginning of each epoch
        #TODO make sure that this works as intended
        #np.random.shuffle(merged)
        #data, labels = np.split(merged, [data.shape[1]], axis=1)

        for i in range(0, len(data), batchsize):
        # Add test after each epoch (also easy)
            my = i
            bs = batchsize
            if i + batchsize >= len(data):
                bs = i + batchsize - len(data)
            V[0] = np.array(data[my:my+bs]).T
            t = np.array(labels[my:my+bs]).T
            #t = t.reshape(bs,t.shape[1])
            
            propagate_forward(V,O,w, bs)
            propagate_backwards(V,O,w,t,learning_rate, bs)
        #print(w[0])
        C = calculate_classification_error(V,w,O,test_data, test_labels)
        #print(w[0])
        print(C)
        

def calculate_classification_error(V,w,O, test_data, test_labels):
    inner_sum = 0
    pval = len(test_data)
    for my in range(pval):
        V[0] = np.array(test_data[my])
        t_my = test_labels[my]
        #Propagate Forward
        propagate_forward(V,O,w,1)
        if (esign(V[-1][0][0]) != t_my[0]):
            inner_sum +=1
            #print("error")
    return inner_sum/pval



def transformlabels(labels, n):
    for i,label in enumerate(labels):
        temp = np.zeros((n,1))
        temp[label] = 1
        labels[i] = temp

"""
folder_name = "/home/edin/uni/ANN/ex-3/cifar-10-batches-py/"
batch_1 = unpickle(folder_name+"data_batch_1")
batch_2 = unpickle(folder_name+"data_batch_2")
batch_3 = unpickle(folder_name+"data_batch_3")
batch_4 = unpickle(folder_name+"data_batch_4")
batch_5 = unpickle(folder_name+"data_batch_5")

data = batch_1[b'data']
data = np.append(data,batch_2[b'data'], axis=0)
data = np.append(data,batch_3[b'data'], axis=0)
data = np.append(data,batch_4[b'data'], axis=0)
#data = np.append(data,batch_5[b'data'], axis=0)
labels = batch_1[b'labels']
#Since labels are only a normal array
labels += batch_2[b'labels']
labels += batch_3[b'labels']
labels += batch_4[b'labels']
#labels += batch_5[b'labels']
data = center_data(data)
transformlabels(labels,10)
labels = np.array(labels)
labels = labels.astype(np.uint8)
labels = np.array(labels).reshape(40000,10)
test_batch = unpickle(folder_name+"test_batch")
test_data = batch_5[b'data']
test_labels = batch_5[b'labels']

transformlabels(test_labels,10)
test_labels = np.array(test_labels)

test_labels = test_labels.astype(np.uint8)
"""
#IT DOES NOT WORK
#POSSIBLE PROBLEMS:
# Permutation is not correct
# -> Add assert statements or smth
# Centering Function is fucked up
# Batch processing is fucked up
# -> Test with Aufgabe 2.1 and see if it works

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


minibatch_size = 1
learning_rate = 0.03
#learn(learning_rate,data,labels,test_data, test_labels, minibatch_size, 3072,10)
learn(learning_rate,training_set[:,:2],training_set[:,2].reshape((10000,1)),validation_set[:,:2], training_set[:,2].reshape((10000,1)), minibatch_size, 2,20,10,1)




