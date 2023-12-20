import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression


pd.options.mode.chained_assignment = None

df_train = pd.read_csv(fr'data\train_final.csv', na_values='?')
df_test = pd.read_csv(fr'data\test_final.csv', na_values='?')

cats = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

for i in cats:
    df_train[i] = df_train[i].astype('category')
    df_test[i] = df_test[i].astype('category')

y_train = df_train['income>50K']
df_train_X = df_train.drop('income>50K', axis=1)

ID_test = df_test['ID']
df_test_X = df_test.drop('ID', axis=1)

print("Data successfuly imported.\n")

# List of variables that contain missing values
missing_vars = ['workclass', 'occupation', 'native.country']

print("Imputing Missing Data.\n")

# For each variable with missing data
for i in range(3):
    pre_train_X = df_train_X.copy()
    pre_test_X = df_test_X.copy()

    current_var = missing_vars[i]

    # First, remove the other two variables with missing values from the datasets

    for j in missing_vars:
        if j != current_var:
            pre_train_X.drop(j, axis=1, inplace=True)
            pre_test_X.drop(j, axis=1, inplace=True)

    # Second, split the datasets into one with the missing data and the other without

    train_missing = pre_train_X[pre_train_X[current_var].isna()]
    train_notmissing = pre_train_X[~pre_train_X[current_var].isna()]
    
    test_missing = pre_test_X[pre_test_X[current_var].isna()]
    test_notmissing = pre_test_X[~pre_test_X[current_var].isna()]

    # Third, convert target (missing_var) to integer values only for the sets without missing data

    train_labels, train_uniques = pd.factorize(train_notmissing[current_var])
    test_labels, test_uniques = pd.factorize(test_notmissing[current_var])

    train_notmissing['target'] = train_labels + 1
    train_notmissing.drop(current_var, axis=1, inplace=True)
    
    test_notmissing['target'] = test_labels + 1
    test_notmissing.drop(current_var, axis=1, inplace=True)

    # Fourth, remove the target from the set with missing values. These will be predicted with the logistic regression multiclass classifier.
    train_missing.drop(current_var, axis=1, inplace=True)
    test_missing.drop(current_var, axis=1, inplace=True)

    # To complete the data preparation, we use one-hot-encoding for the remaining categorical variables

    train_notmissing_processed = pd.get_dummies(train_notmissing, "is")
    train_missing_processed = pd.get_dummies(train_missing, "is")
    
    test_notmissing_processed = pd.get_dummies(test_notmissing, "is")
    test_missing_processed = pd.get_dummies(test_missing, "is")

    # Create input data, model, and data used to predict missing values
    train_X = train_notmissing_processed.drop('target', axis=1).to_numpy()
    train_y = train_notmissing_processed['target'].to_numpy()
    train_X_missing = train_missing_processed.to_numpy()

    model = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=1000)
    
    model.fit(train_X, train_y)
    train_mv = model.predict(train_X_missing)

    train_imputed_mv = np.array(train_uniques[train_mv])
    df_train_X.loc[df_train_X[current_var].isna(), [current_var]] = train_imputed_mv
    
    # Repeat for test set
    test_X = test_notmissing_processed.drop('target', axis=1).to_numpy()
    test_y = test_notmissing_processed['target'].to_numpy()
    test_X_missing = test_missing_processed.to_numpy()
    
    model.fit(test_X, test_y)
    test_mv = model.predict(test_X_missing)
    
    test_imputed_mv = np.array(test_uniques[test_mv])
    df_test_X.loc[df_test_X[current_var].isna(), [current_var]] = test_imputed_mv


X_train = pd.get_dummies(df_train_X, "is")
X_test = pd.get_dummies(df_test_X, "is")

newcol = [0] * 25000

index_no = X_test.columns.get_loc('is_Holand-Netherlands')

X_train.insert(loc=index_no,
               column = 'is_Holand-Netherlands', 
               value = newcol)

X_train = X_train.astype(dtype=float)
X_test = X_test.astype(dtype=float)

for i in range(6):
    orig = X_train.iloc[:,i].to_numpy()
    normalized = (orig - np.mean(orig)) / np.std(orig)
    X_train.iloc[:,i] = normalized
    
    orig = X_test.iloc[:, i].to_numpy()
    normalized = (orig - np.mean(orig)) / np.std(orig)
    X_test.iloc[:,i] = normalized

print("Imputation Complete.\n")

X = X_train.to_numpy()
y = y_train.to_numpy()

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)


print("Beginning 5-Fold Cross-Validation Procedure.\n")

in_width = len(X_train.columns)

nets = []

# 2 hidden 75 width ReLU
H2_W75_ReLU = nn.Sequential(
    nn.Linear(in_width, 75),
    nn.ReLU(),
    nn.Linear(75, 75),
    nn.ReLU(),
    nn.Linear(75, 1),
    nn.Sigmoid()
)

nets.append(H2_W75_ReLU)

# 2 hidden 75 width ELU

H2_W75_ELU = nn.Sequential(
    nn.Linear(in_width, 75),
    nn.ELU(),
    nn.Linear(75, 75),
    nn.ELU(),
    nn.Linear(75, 1),
    nn.Sigmoid()
)

nets.append(H2_W75_ELU)

# 2 hidden 100 width ReLU

H2_W100_ReLU = nn.Sequential(
    nn.Linear(in_width, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 1),
    nn.Sigmoid()
)

nets.append(H2_W100_ReLU)

# 2 hidden 100 width ELU

H2_W100_ELU = nn.Sequential(
    nn.Linear(in_width, 100),
    nn.ELU(),
    nn.Linear(100, 100),
    nn.ELU(),
    nn.Linear(100, 1),
    nn.Sigmoid()
)

nets.append(H2_W100_ELU)

# 3 hidden 75 width ReLU

H3_W75_ReLU = nn.Sequential(
    nn.Linear(in_width, 75),
    nn.ReLU(),
    nn.Linear(75, 75),
    nn.ReLU(),
    nn.Linear(75, 75),
    nn.ReLU(),
    nn.Linear(75, 1),
    nn.Sigmoid()
)

nets.append(H3_W75_ReLU)

# 3 hidden 75 width ELU

H3_W75_ELU = nn.Sequential(
    nn.Linear(in_width, 75),
    nn.ELU(),
    nn.Linear(75, 75),
    nn.ELU(),
    nn.Linear(75, 75),
    nn.ELU(),
    nn.Linear(75, 1),
    nn.Sigmoid()
)

nets.append(H3_W75_ELU)

# 3 hidden 100 width ReLU

H3_W100_ReLU = nn.Sequential(
    nn.Linear(in_width, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 1),
    nn.Sigmoid()
)

nets.append(H3_W100_ReLU)

# 3 hidden 100 width ELU

H3_W100_ELU = nn.Sequential(
    nn.Linear(in_width, 100),
    nn.ELU(),
    nn.Linear(100, 100),
    nn.ELU(),
    nn.Linear(100, 100),
    nn.ELU(),
    nn.Linear(100, 1),
    nn.Sigmoid()
)

nets.append(H3_W100_ELU)


net_descs = []

for i in ['2 Hidden', '3 Hidden']:
    for j in ['75', '100']:
        for k in ['ReLU', 'ELU']:
            string = "NN w/ " + i + " Layers, Hidden Layer Size " + j + ", " + k + " Activation"
            net_descs.append(string)

k=5
cv_folds = KFold(k)
epochs = 50
batchsz = 200
rate = .01

print("Hyperparameters Held Constant For Cross Validation:")
print(f"Learning Rate = {rate}")
print(f"Epochs = {epochs}")
print(f"Batch Size = {batchsz}\n")

for j in range(len(nets)):
    
    acc_score = []

    for train_index , test_index in cv_folds.split(X_train):
        xtr , xcv = X_train.iloc[train_index,:].to_numpy(),X_train.iloc[test_index,:].to_numpy()
        ytr , ycv = y_train[train_index].to_numpy() , y_train[test_index].to_numpy()
        
        # Convert Numpy to Tensor
        
        xtr = torch.tensor(xtr, dtype=torch.float32).clone().detach()
        ytr = torch.tensor(ytr, dtype=torch.float32).reshape(-1, 1).clone().detach()
        
        xcv = torch.tensor(xcv, dtype=torch.float32).clone().detach()
        ycv = torch.tensor(ycv, dtype=torch.float32).reshape(-1, 1).clone().detach()
        
        # Train the model, predict on validation set, and append prediction accuracy to acc_score list
        
        # Define network
        network = nets[j]
        
        # Define loss and optimizer
        loss_fn   = nn.BCELoss()  # binary cross entropy
        optimizer = optim.Adam(network.parameters(), lr=rate)
        
        # Train network with the given number of epochs and batch size
        for epoch in range(epochs):
            
            shuffle_idx = torch.randperm(xtr.size()[0])

            X_shuffled = xtr[shuffle_idx].clone().detach()
            y_shuffled = ytr[shuffle_idx].clone().detach()
            
            for i in range(0, len(xtr), batchsz):
                Xbatch = X_shuffled[i:i+batchsz]
                y_pred = network(Xbatch)
                ybatch = y_shuffled[i:i+batchsz]
                loss = loss_fn(y_pred, ybatch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Print loss on every tenth epoch
            # if epoch % 10 == 0:
            #     print(f'Finished epoch {epoch}, latest loss {loss}')
                
        with torch.no_grad():
            cv_pred = network(xcv)
        accuracy = (cv_pred.round() == ycv).float().mean()
        
        acc_score.append(accuracy / k)

    print(net_descs[j])
    print(f"Cross-Val Accuracy: {round(float(sum(acc_score)), 4)}\n")

print("Cross-Validation Complete.\n")

in_width = X.shape[1]
h_width = in_width

n_epochs = 50
batch_size = 100
rate = .01

# define the model
network = nn.Sequential(
    nn.Linear(in_width, h_width),
    nn.ReLU(),
    nn.Linear(h_width, h_width),
    nn.ReLU(),
    nn.Linear(h_width, h_width),
    nn.ReLU(),
    nn.Linear(h_width, 1),
    nn.Sigmoid()
)

print("Training Final Model\n")
print(f"Input Layer Size = {in_width}")
print(f"Hidden Layer Size = {h_width}")
print("Number Hidden Layers = 3")
print("Hidden Activations = ReLU")
print("Output Activation = Sigmoid")
print(f"Learning Rate = {rate}")
print(f"Epochs = {n_epochs}")
print(f"Batch Size = {batch_size}\n")

# train the model
loss_fn   = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(network.parameters(), lr=rate)

for epoch in range(n_epochs):
    
    shuffle_idx = torch.randperm(X.size()[0])

    X_shuffled = X[shuffle_idx].clone().detach()
    y_shuffled = y[shuffle_idx].clone().detach()
    
    
    for i in range(0, len(X), batch_size):
        Xbatch = X_shuffled[i:i+batch_size]
        y_pred = network(Xbatch)
        ybatch = y_shuffled[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Finished epoch {epoch + 1}, Current Loss = {loss}")

print("\nModel Successfully Trained.")

with torch.no_grad():
    y_pred = network(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Train Accuracy = {accuracy}\n")

Xts = X_test.to_numpy()

Xts = torch.tensor(Xts, dtype=torch.float32)

predictions = network(Xts)

print("Test Predictions Successfully Computed.\n")

out_df = pd.DataFrame(ID_test)

out_df['Prediction'] = predictions.detach().numpy()

out_df.to_csv(fr'predictions\predictions.csv', index=False)

print("Prediction Output Saved To 'predictions' Folder, final.py Complete.")