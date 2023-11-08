import pandas as pd
import numpy as np

def perc_standard(data, epochs, rate = .1):
    w = np.full(len(data[0]) - 1, 0)
    for i in range(epochs):
        shuffled = np.copy(data)
        np.random.shuffle(shuffled)
        for row in shuffled:
            if row[-1] * np.dot(w, row[0:-1]) <= 0:
                w = w + (rate * row[-1] * row[0:-1])
    return w

def perc_voted(data, epochs, rate = .1):
    w_vectors = []
    counts = []
    c = 0
    w = np.full(len(data[0]) - 1, 0)
    
    for i in range(epochs):
        for row in data:
            if row[-1] * np.dot(w, row[0:-1]) <= 0:
                w_vectors.append(w)
                counts.append(c)
                w = w + (rate * row[-1] * row[0:-1])
                c = 1
            else:
                c += 1

    w_vectors.pop(0)
    counts.pop(0)
    
    w_vectors_np = np.array(w_vectors)
    counts_np = np.array(counts)
        
    return w_vectors_np, counts_np


def perc_averaged(data, epochs, rate = .1):
    w = np.full(len(data[0]) - 1, 0)
    a = np.full(len(data[0]) - 1, 0)
    
    for i in range(epochs):
        for row in data:
            if row[-1] * np.dot(w, row[0:-1]) <= 0:
                w = w + (rate * row[-1] * row[0:-1])
            a = a + w
    return a

cols = ['variance', 'skewness', 'kurtosis', 'entropy', 'label']
train = pd.read_csv(fr'data\train.csv', names=cols)
test = pd.read_csv(fr'data\test.csv', names=cols)
train['label'].replace(0, -1, inplace=True)
test['label'].replace(0, -1, inplace=True)
train.insert(0, 'bias', 1)
test.insert(0, 'bias', 1)

train_np = train.to_numpy()
test_np = test.to_numpy()

train_np_X = train.drop('label', axis=1).to_numpy()
train_np_Y = train['label'].to_numpy()

test_np_X = test.drop('label', axis=1).to_numpy()
test_np_Y = test['label'].to_numpy()


print("Standard Perceptron\n")

standard_w = perc_standard(train_np, 10)

standard_w1 = np.reshape(standard_w, (5,1))

preds1 = np.matmul(test_np_X, standard_w1)
predictions1 = []

for i in preds1:
    if i <= 0:
        predictions1.append(-1)
    else:
        predictions1.append(1)

predictions1 = np.array(predictions1)

print("average test error: ")
print(1 - np.mean(predictions1 == test_np_Y))
print()

print("learned weight vector: ")
print(standard_w)
print()




ws, cs = perc_voted(train_np, 10)

preds_sum = np.full(len(test_np), 0)

for i in range(len(ws)):

    current_reshaped_w = np.reshape(ws[i], (5,1))
    current_count = cs[i]

    preds2 = np.matmul(test_np_X, current_reshaped_w)

    predictions2 = []

    for j in preds2:
        if j <= 0:
            predictions2.append(-1)
        else:
            predictions2.append(1)
            
    predictions2 = np.array(predictions2)

    predictions2 = predictions2 * current_count

    preds_sum = preds_sum + predictions2

final_preds = []

for i in preds_sum:
    if i <= 0:
        final_preds.append(-1)
    else:
        final_preds.append(1)
        
final_predictions = np.array(final_preds)

print("Voted Perceptron\n")

print("average test error: ")
print(1 - np.mean(final_predictions == test_np_Y))
print()

print("learned weight vectors: ")

df_latex = pd.DataFrame(np.round(ws, 2), columns= train.drop('label', axis=1).columns)
df_latex['#_Correct'] = cs

out = df_latex.to_numpy()

np.round(out, 2)

for i in range(len(out)):
    print(f"{i}: {out[i]}")
print()


print("Averaged Perceptron\n")



weightss = perc_averaged(train_np, 10)

weightss1 = np.reshape(weightss, (5,1))

predss = np.matmul(test_np_X, weightss1)
predictionss = []

for i in predss:
    if i <= 0:
        predictionss.append(-1)
    else:
        predictionss.append(1)

predictionss = np.array(predictionss)

print("average test error: ")
print(1 - np.mean(predictionss == test_np_Y))
print()

print("learned weight vectors: ")
print(weightss)
print()
