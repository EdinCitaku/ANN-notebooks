import numpy as np
#Create p random patterns
def g(b, beta):
    return 1/(1+np.exp(-2*beta*b))

def esign(x):

    if(x == 0):
        return 1
    else:
        return np.sign(x)


def estimate_m(T, p, N, beta):

    patterns = []
    for i in range(p):
        patterns.append(np.random.choice([-1,1],N))
        #Create weights matrix according to hebbs rule
    weights = patterns[0][:,None]*patterns[0]
    for el in patterns[1:]:
        weights = weights + el[:,None]*el
    weights = weights/N
    np.fill_diagonal(weights,0)
    S = patterns[0].copy()
    m = 0
    for i in range(T):
        chosen_i = np.random.choice(range(N))
        probability = g(np.dot(weights[chosen_i], S),beta)
        S[chosen_i] = esign(np.random.choice([+1,-1],p=[probability, 1-probability]))
        m = m + np.dot(S,patterns[0])/N
    m = m/T
    return m
#Exercise A
m_average = 0
for i in range(100):
    m_average+= estimate_m(100000, 45, 200, 2)
m_average = m_average/100
print(m_average)

#Exercise B
m_average = 0
for i in range(100):
    m_average+= estimate_m(100000, 45, 200, 2)
m_average = m_average/100
print(m_average)