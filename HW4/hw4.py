import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

cols = ['variance', 'skewness', 'kurtosis', 'entropy', 'label']
train = pd.read_csv(fr'data\train.csv', names=cols)
test = pd.read_csv(fr'data\test.csv', names=cols)
train['label'].replace(0, -1, inplace=True)
test['label'].replace(0, -1, inplace=True)

train_np_X1 = train.drop('label', axis=1).to_numpy()
train_np_Y1 = train['label'].to_numpy()

test_np_X1 = test.drop('label', axis=1).to_numpy()
test_np_Y1 = test['label'].to_numpy()

train.insert(0, 'bias', 1)
test.insert(0, 'bias', 1)

train_np = train.to_numpy()
test_np = test.to_numpy()

train_np_X = train.drop('label', axis=1).to_numpy()
train_np_Y = train['label'].to_numpy()

test_np_X = test.drop('label', axis=1).to_numpy()
test_np_Y = test['label'].to_numpy()

# SVM Stochastic Sub Gradient Descent w/ First Step Size Schedule
def svm_sgd1(data, epochs = 100, rate = .01, a = 1, tradeoff = .5):
    w = np.full(len(data[0]) - 1, 0)
    t = 0
    for i in range(epochs):
        shuffled = np.copy(data)
        np.random.shuffle(shuffled)
        for row in shuffled:
            
            if row[-1] * np.dot(w, row[0:-1]) <= 1:
                w0 = np.copy(w)
                w0[0] = 0
                w = w - (rate * w0) + (rate * tradeoff * len(data) * row[-1] * row[0:-1])
            else:
                w0 = np.copy(w) * (1-rate)
                w0[0] = w[0]
                w = np.copy(w0)
            
            rate = rate / (1 + (rate * t / a))
            t += 1

    return w

# SVM Stochastic Sub Gradient Descent w/ Second Step Size Schedule
def svm_sgd2(data, epochs = 100, rate = .01, tradeoff = .5):
    w = np.full(len(data[0]) - 1, 0)
    t = 0
    for i in range(epochs):
        shuffled = np.copy(data)
        np.random.shuffle(shuffled)
        for row in shuffled:
            
            if row[-1] * np.dot(w, row[0:-1]) <= 1:
                w0 = np.copy(w)
                w0[0] = 0
                w = w - (rate * w0) + (rate * tradeoff * len(data) * row[-1] * row[0:-1])
            else:
                w0 = np.copy(w) * (1-rate)
                w0[0] = w[0]
                w = np.copy(w0)
            
            rate /= (1 + t)
            t += 1

    return w

# SVM Linear Dual Optimization
def dual_SVM(X, Y, C = .5, a0 = None):
    
    if type(a0) != 'numpy.ndarray':
        a0 = np.zeros((len(X),))
    
    def objective(alphas, x, y):
        
        x1 = np.copy(x)
        
        for i in range(len(x)):
            x1[i] = y[i] * alphas[i] * x1[i]
            
        gram = np.matmul(x1, np.transpose(x1))
        
        return (np.sum(gram) / 2) - np.sum(alphas)

    def constrain(alphas):
        return np.dot(alphas, Y)

    bound = (0, C)
    bnds = list()

    for i in range(len(X)):
        bnds.append(bound)

    cons = ({'type': 'eq' ,'fun': constrain})
    
    sol = minimize(fun=objective, args=(X, Y), x0=a0, method='SLSQP', constraints=cons, bounds=bnds, tol=.1)
    
    weights = np.zeros(X.shape[1])
    
    for i in range(len(X)):
        if sol.x[i] != 0:
            weights = weights + (sol.x[i] * Y[i] * X[i])

    bias = np.sum(Y - np.matmul(X, weights))
    
    bias = bias / len(X)
    
    return weights, bias

# SVM Nonlinear RBF Dual Optimization
def dual_SVM_nonlin(X, Y, test_X, test_Y, C = .5, gamma = .1, a0=None):
    
    def objective(alphas, x, y):
        pairwise_sq_dists = squareform(pdist(x, 'sqeuclidean'))
        K = np.exp(-pairwise_sq_dists / gamma)

        for i in range(len(x)):
            K[i,:] *= (y[i] * alphas[i])
            K[:,i] *= (y[i] * alphas[i])
            
        return (np.sum(K) / 2) - np.sum(alphas)
    
    def constrain(alphas):
        return np.dot(alphas, Y)
    
    def rbf(x1, x2):
        return np.exp(-((np.linalg.norm(x1 - x2)**2) / gamma))
        
    
    bound = (0, C)
    bnds = list()

    for i in range(len(X)):
        bnds.append(bound)

    cons = ({'type': 'eq' ,'fun': constrain})
    
    sol = minimize(fun=objective, args=(X, Y), x0=a0, method='SLSQP', constraints=cons, bounds=bnds, tol=1)
    
    alpha_star = sol.x
    
    num_sv = 0
    for i in alpha_star:
        if i > 0:
            num_sv += 1
    
    bias = 0
    
    for j in range(len(X)):
        inner_sum = 0
        for i in range(len(X)):
            if alpha_star[i] != 0:
                inner_sum += alpha_star[i] * Y[i] * rbf(X[i], X[j])
                
        bias += Y[j] - inner_sum
    
    train_preds = []
    
    for j in range(len(X)):
        current_pred = 0
        for i in range(len(X)):
            current_pred += alpha_star[i] * Y[i] * rbf(X[i], X[j])

        if current_pred + bias > 0:
            train_preds.append(1)
        else:
            train_preds.append(-1)
    
    test_preds = []
    
    for j in range(len(test_X)):
        current_pred = 0
        for i in range(len(X)):
            current_pred += alpha_star[i] * Y[i] * rbf(X[i], test_X[j])
        
        if current_pred + bias > 0:
            test_preds.append(1)
        else:
            test_preds.append(-1)
            
    train_predictions = np.array(train_preds)
    test_predictions = np.array(test_preds)
    
    train_err = 1 - np.mean(train_predictions == Y)
    test_err = 1 - np.mean(test_predictions == test_Y)
    
    return alpha_star, train_err, test_err, num_sv

l = lambda x: 1 if x > 0 else -1
convert = np.vectorize(l)

model_params1 = []
model_params2 = []

train_er1 = []
train_er2 = []

test_er1 = []
test_er2 = []

cvals = ["100/872", "500/872", "700/872"]

for i in cvals:
    num,den = i.split('/')
    c = int(num)/int(den)
    
    weights1 = svm_sgd1(train_np, tradeoff=c)
    weights2 = svm_sgd2(train_np, tradeoff=c)
    
    model_params1.append(weights1)
    model_params2.append(weights2)
    
    train_wtx1 = np.matmul(train_np_X, weights1)
    train_err1 = 1 - np.mean(convert(train_wtx1) == train_np_Y)
    train_er1.append(train_err1)
    
    train_wtx2 = np.matmul(train_np_X, weights2)
    train_err2 = 1 - np.mean(convert(train_wtx2) == train_np_Y)
    train_er2.append(train_err2)
    
    test_wtx1 = np.matmul(test_np_X, weights1)
    test_err1 = 1 - np.mean(convert(test_wtx1) == test_np_Y)
    test_er1.append(test_err1)
    
    test_wtx2 = np.matmul(test_np_X, weights2)
    test_err2 = 1 - np.mean(convert(test_wtx2) == test_np_Y)
    test_er2.append(test_err2)


# 2.2.a

print("Problem 2.2.a\n")
for i in range(len(cvals)):
    print(f"SVM Primal: C={cvals[i]}")
    print(f"Parameters: {np.round(model_params1[i], 4)}")
    print(f"Train Error = {np.round(train_er1[i], 4)}")
    print(f"Test Error = {np.round(test_er1[i], 4)}\n")

# 2.2.b

print("\nProblem 2.2.b\n")
for i in range(len(cvals)):
    print(f"SVM Primal: C={cvals[i]}")
    print(f"Parameters: {np.round(model_params2[i], 4)}")
    print(f"Train Error = {np.round(train_er2[i], 4)}")
    print(f"Test Error = {np.round(test_er2[i], 4)}\n")

# 2.2.c

print("\nProblem 2.2.c\n")
for i in range(len(cvals)):
    print(f"SVM Primal: C={cvals[i]}")
    print(f"Model Parameters Difference: {np.round(model_params1[i] - model_params2[i], 4)}")
    print(f"Train Error Difference = {-np.round(train_er1[i] - train_er2[i], 4)}")
    print(f"Test Error Difference = {-np.round(test_er1[i] - test_er2[i], 4)}\n")

# 2.3.a

print("\nProblem 2.3.a\n")
for i in ["100/872", "500/872", "700/872"]:
    num,den = i.split('/')
    c = int(num)/int(den)
    
    weights, bias = dual_SVM(train_np_X1, train_np_Y1, C = c,  a0 = np.zeros(872))
    
    wtx_train = np.matmul(train_np_X1, weights)
    b_train = np.full(len(train_np_X1), bias)
    preds_train = wtx_train + b_train
    train_error = 1 - np.mean(convert(preds_train) == train_np_Y1)
    
    wtx_test = np.matmul(test_np_X1, weights)
    b_test = np.full(len(test_np_X1), bias)
    preds_test = wtx_test + b_test
    test_error = 1 - np.mean(convert(preds_test) == test_np_Y1)
    
    print(f"Dual SVM C={i}:")
    print(f"Weights = {np.round(weights, 4)}")
    print(f"Bias = {np.round(bias, 4)}")
    print(f"Train Error = {np.round(train_error, 4)}")
    print(f"Test Error = {np.round(test_error, 4)}\n")

# 2.3.b
print("\nProblem 2.3.b\n")
dicts = []
for i in ["100/872", "500/872", "700/872"]:
    for g in [.1, .5, 1, 5, 100]:
        num,den = i.split('/')
        c = int(num)/int(den)
        
        optimal_a, train_er, test_er, num_support_vecs = dual_SVM_nonlin(train_np_X1, train_np_Y1, test_np_X1, test_np_Y1, C = c, gamma = g, a0=np.zeros(len(train_np_X)))
        
        print(f"Nonlinear Dual SVM: C={i}, Gamma={g}")
        print(f"Train Error = {np.round(train_er, 4)}")
        print(f"Test Error = {np.round(test_er, 4)}\n")
        
        temp_dict = {
            "C" : i,
            "Gamma" : g,
            "Train Error" : train_er,
            "Test Error" : test_er,
            "# Support Vectors" : num_support_vecs,
            "Alphas" : optimal_a
        }
        
        dicts.append(temp_dict)

# 2.3.c
print("\nProblem 2.3.c\n")
print(f"Nonlinear Dual SVM:")
for i in dicts:
    print(f"C={i['C']}, Gamma={i['Gamma']}")
    print(f"Number of Support Vectors = {i['# Support Vectors']}\n")

tuples = []

clevel = "500/872"

for i in dicts:
    if i['C'] == clevel:
        tuples.append((i['Gamma'], i['Alphas']))

print(f"Nonlinear SVM Dual: C={clevel}")
for i in range(len(tuples)- 1):
    als_prev = tuples[i][1]
    als_next = tuples[i+1][1]
    
    num_overlap = 0
    
    for j in range(len(als_prev)):
        if als_prev[j] > 0 and als_next[j] > 0:
            num_overlap +=1
    
    print(f"Comparing Nonlinear SVM w/ gamma={tuples[i][0]} & gamma={tuples[i+1][0]}:")
    print(f"# Overlapping SVs = {num_overlap}\n")
