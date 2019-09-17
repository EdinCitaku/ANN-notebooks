import numpy as np

def calculate_instance( n, p):
    #Create p random patterns
    patterns = []
    
    for i in range(p):
        patterns.append(np.random.choice([-1,1],n))
    #Create weights matrix according to hebbs rule
    weights = patterns[0][:,None]*patterns[0]
    for el in patterns[1:]:
        weights = weights + el[:,None]*el
    weights = np.true_divide(weights, n)
    
    #Fill diagonal with zeroes
    np.fill_diagonal(weights,0)
    #Feed random pattern as input and test if an error occurs
    S1 = patterns[0].copy().T
    chosen_i = np.random.choice(range(n))
    S_i = esign(np.dot(weights[chosen_i], S1))
    S1[chosen_i] = S_i
    #breakpoint()
    return np.array_equal(S1,patterns[0].T)

def esign(x):

    if(x == 0):
        return 1
    else:
        return np.sign(x)

p = [12, 24, 48, 70, 100, 120]
N = 120
solve = [0,0]
I = 100000
for p_i in p:
    for i in range(I):
        ret = calculate_instance(N, p_i)
        if ret:
            solve[0]+=1
        else:
            solve[1]+=1
    p_error = float(solve[1]/I) 
    print(f"Number of patterns: {p}, P_error(t=1): {p_error} ")
