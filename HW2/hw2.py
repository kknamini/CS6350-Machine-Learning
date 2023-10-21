print("\nImporting packages & scripts.")

import numpy as np
import pandas as pd
import dt as dt1
from matplotlib import pyplot as plt

print("Importing complete.\n")

print("Reading & processing bank data.")

column_names = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"]
d_types = ["num", "cat", "cat", "cat", "cat", "num", "cat", "cat", "cat", "num", "cat", "num", "num", "num", "num", "cat", "cat"]

bank_train = pd.read_csv(r"data\bank-train.csv", names=column_names)
bank_train, median_values = dt1.convert_numerical(bank_train, d_types)
bank_train["y"].replace('yes', 1, inplace=True)
bank_train["y"].replace('no', -1, inplace=True)

bank_test = pd.read_csv(r"data\bank-test.csv", names = column_names)
bank_test = dt1.convert_numerical(bank_test, d_types, median_values)[0]
bank_test["y"].replace('yes', 1, inplace=True)
bank_test["y"].replace('no', -1, inplace=True)

attr_names = bank_train.columns.to_list()
attr_names.remove("y")
attr_values = [["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                "blue-collar","self-employed","retired","technician","services"],
               ["married","divorced","single"],
               ["unknown","secondary","primary","tertiary"],
               ["yes","no"],
               ["yes","no"],
               ["yes","no"],
               ["unknown","telephone","cellular"],
               ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
               ["unknown","other","failure","success"],
               [0, 1],
               [0, 1],
               [0, 1],
               [0, 1],
               [0, 1],
               [0, 1],
               [0, 1]]

Attrs_bank = dict(zip(attr_names, attr_values))

y = bank_train.pop('y')

bank_train['y'] = y

train_np = bank_train.to_numpy()

train_X_np = np.delete(train_np, -1, axis=1)

train_Y = train_np[:,-1].tolist()

y = bank_test.pop('y')

bank_test['y'] = y

test_np = bank_test.to_numpy()

test_X_np = np.delete(test_np, -1, axis=1)

test_Y = test_np[:,-1].tolist()

print("Bank data preparation complete.")

print("\n###########################################################\n")

#################################################################################

print("Problem 2.2.a\n")
print("Creating plot 1: Train & Test Error for Adaboost Decision Stumps")

# Create lists to hold votes and weak classifiers
vote_stump_pairs = []

# Choose number of iterations/weak classifiers, T
T = 500

# Initialize weight vector
# D = np.full(len(train_X_np), 1/len(train_X_np))

D = [1 / len(train_X_np)] * len(train_X_np)

# Run T times
for i in range(T):
    # Learn weak classifier
    stump = dt1.DecisionTree()
    stump.root = dt1.ID3(train_X_np, train_Y, Attrs_bank, "H", max_depth=1, weights = list(D))

    # Compute the weak classifier's vote
    predict_train = []
    for index in range(len(train_X_np)):
        predict_train.append(D[index] * (stump.predict(train_X_np[index], Attrs_bank) == train_Y[index]))
    
    vote = np.log(sum(predict_train) / (1 - sum(predict_train))) / 2

    # Update each weight value in the weights vector
    for index in range(len(train_X_np)):
        D[index] *= np.exp(-vote * train_Y[index] * stump.predict(train_X_np[index], Attrs_bank))

    D = D / sum(D)
    
    # Append the current vote and classifier to their respective list
    vote_stump_pairs.append((vote, stump))

# Reset Node IDs
dt1.Node.reset_id_gen()

# Vary the number of iterations/stumps used in Adaboost
num_stumps = [10, 25, 50, 100, 300, T]

# Create nested lists to hold train & test accuracy for each number of iterations above
ada_train_acc = []
ada_test_acc = []

for i in num_stumps:
    ada_train_acc.append([])
    ada_test_acc.append([])

# For each row in the training data
for index in range(len(train_X_np)):
    
    # Initialize the prediction for a single row
    row_pred = 0
    
    # For each iteration/stump
    for i in range(T):
        
        # Modify the predicted value using the current vote and stump prediction
        row_pred += vote_stump_pairs[i][0] * vote_stump_pairs[i][1].predict(train_X_np[index], Attrs_bank)

        # If the current iteration is in the list containing the varied number of iterations
        if i+1 in num_stumps:
            
            # If the current prediction after i + 1 iterations is positive
            if row_pred > 0:
                
                # Append True to the appropriate nested list if the actual label is also positive, otherwise False
                ada_train_acc[num_stumps.index(i+1)].append(1 == train_Y[index])
            
            # If the current prediction after i + 1 iterations is negative
            else:
                
                # Append True to the appropriate nested list if the actual label is also negative, otherwise False
                ada_train_acc[num_stumps.index(i+1)].append(-1 == train_Y[index])
                

# For each row in the testing data
for index in range(len(test_X_np)):
    
    # Initialize the prediction for a single row
    row_pred = 0
    
    # For each iteration/stump
    for i in range(T):
        
        # Modify the predicted value using the current vote and stump prediction
        row_pred += vote_stump_pairs[i][0] * vote_stump_pairs[i][1].predict(test_X_np[index], Attrs_bank)
        
        # If the current iteration is in the list containing the varied number of iterations
        if i+1 in num_stumps:
            
            # If the current prediction after i + 1 iterations is positive
            if row_pred > 0:
                
                # Append True to the appropriate nested list if the actual label is also positive, otherwise False
                ada_test_acc[num_stumps.index(i+1)].append(1 == test_Y[index])
            
            # If the current prediction after i + 1 iterations is negative
            else:
                
                # Append True to the appropriate nested list if the actual label is also negative, otherwise False
                ada_test_acc[num_stumps.index(i+1)].append(-1 == test_Y[index])

# Create list and populate with training accuracies for plot
Y1 = []
for i in ada_train_acc:
    Y1.append(1 - (sum(i) / len(i)))
    
# Create list and populate with testing accuracies for plot
Y2 = []
for i in ada_test_acc:
    Y2.append(1 - (sum(i) / len(i)))

# Populate X axis with each varied number of iterations/stumps
X = num_stumps

# Create line plot for 2.2.a
plt.plot(X, Y1, marker = 'o')
plt.plot(X, Y2, marker = 'o')
plt.title("Train & Test Error for Adaboost Decision Stumps")
plt.xlabel("# Iterations/Stumps")
plt.ylabel("Error %")
plt.legend(['Train', "Test"])
plt.savefig(rf'plots\22a_plot1.png')
plt.close()

print("Plot 1 has been saved to the 'plots' folder.\n")

print("Creating plot 2: Train & Test Error for Individual Decision Stumps")

# Create lists to hold train and test accuracies for all 500 decision stumps
train_err = []
test_err = []

# For each iteration/stump
for i in range(T):
    
    # Create temporary lists for accuracy calculations
    temp_train_acc = []
    temp_test_acc = []
    
    # For each row in the training set
    for index in range(len(train_X_np)):
        
        # Determine if the prediciton is consistent with the data, and append the result into the appropriate temp list
        temp_train_acc.append(vote_stump_pairs[i][1].predict(train_X_np[index], Attrs_bank) == train_Y[index])
    
    # For each row in the testing set
    for index in range(len(test_X_np)):
        
        # Determine if the prediciton is consistent with the data, and append the result into the appropriate temp list
        temp_test_acc.append(vote_stump_pairs[i][1].predict(test_X_np[index], Attrs_bank) == test_Y[index])
    
    # Calculate the current stump's train accuracy and append to the appropriate list
    train_err.append(1 - (sum(temp_train_acc) / len(temp_train_acc)))
    
    # Calculate the current stump's test accuracy and append to the appropriate list
    test_err.append(1 - (sum(temp_test_acc) / len(temp_test_acc)))

# Create histogram plot for 2.2.a
plt.hist([train_err, test_err], color = ['b', 'g'], bins = 5)
plt.legend(['Train', 'Test'], loc = 'upper center')
plt.title("Train & Test Error for Individual Decision Stumps")
plt.xlabel("Error %")
plt.ylabel("# Stumps")
plt.savefig(rf'plots\22a_plot2.png')
plt.close()

print("Plot 2 has been saved to the 'plots' folder.")

print("\n###########################################################\n")

#################################################################################

print("Problem 2.2.b\n")
print("Creating plot 1: Train & Test Error for Bagged Decision Trees")

# Set number of iterations/bagged trees to construct
T = 500

# Create list to hold each bagged tree
bagged_trees = []

# Repeat T times
for i in range(T):
    
    bs_train = train_np[np.random.randint(0,5000,5000),:]
    
    bs_X = np.delete(bs_train, -1, axis=1)
    bs_Y = bs_train[:,-1].tolist()
    
    # Create bagged tree and fit to bootstrap sample
    tree = dt1.DecisionTree()
    tree.root = dt1.ID3(bs_X, bs_Y, Attrs_bank)
    
    # Append bagged tree to list of all bagged trees
    bagged_trees.append(tree)

# Vary the number of trees used in Bagged Trees
num_trees = [10, 25, 50, 100, 200, T]

# Create nested lists to hold test and train accuracy for each number of iterations/trees above
bag_train_acc = []
bag_test_acc = []

for i in num_trees:
    bag_train_acc.append([])
    bag_test_acc.append([])

# For each row in the training set
for index in range(len(train_X_np)):
    
    # Set the current row prediction to zero
    row_pred = 0
    
    # For each bagged tree
    for i in range(len(bagged_trees)):
        
        # Add the current tree's prediction of the current row in training data to the current row prediction value
        row_pred += bagged_trees[i].predict(train_X_np[index], Attrs_bank)
        
        # If the current iteration exists in the list of varied iteration numbers
        if i + 1 in num_trees:
            
            # If the current prediction is positive after i + 1 iterations
            if row_pred > 0:
                
                # Append True to the appropriate nested list if the actual label is also positive, otherwise False
                bag_train_acc[num_trees.index(i+1)].append(1 == train_Y[index])
                
            # If the current prediction after i + 1 iterations is negative
            else:
                
                # Append True to the appropriate nested list if the actual label is also negative, otherwise False
                bag_train_acc[num_trees.index(i+1)].append(-1 == train_Y[index])

# For each row in the testing set
for index in range(len(test_X_np)):
    
    # Set the current row prediction to zero
    row_pred = 0
    
    # For each bagged tree
    for i in range(len(bagged_trees)):
        
        # Add the current tree's prediction of the current row in training data to the current row prediction value
        row_pred += bagged_trees[i].predict(test_X_np[index], Attrs_bank)
        
        # If the current iteration exists in the list of varied iteration numbers
        if i + 1 in num_trees:
            
            # If the current prediction is positive after i + 1 iterations
            if row_pred > 0:
                
                # Append True to the appropriate nested list if the actual label is also positive, otherwise False
                bag_test_acc[num_trees.index(i+1)].append(1 == test_Y[index])
            
            # If the current prediction after i + 1 iterations is negative
            else:
                
                # Append True to the appropriate nested list if the actual label is also negative, otherwise False
                bag_test_acc[num_trees.index(i+1)].append(-1 == test_Y[index])

# Create list and populate with training accuracies for plot
Y1 = []
for i in bag_train_acc:
    Y1.append(1 - (sum(i) / len(i)))

# Create list and populate with testing accuracies for plot
Y2 = []
for i in bag_test_acc:
    Y2.append(1 - (sum(i) / len(i)))

# Populate X axis with each varied number of iterations/bagged trees
X = num_trees

# Create line plot for 2.2.b
plt.plot(X, Y1, marker = 'o')
plt.plot(X, Y2, marker = 'o')
plt.title("Train & Test Error for Bagged Decision Trees")
plt.legend(['Train', "Test"])
plt.xlabel("# Iterations/Bagged Trees")
plt.ylabel("Error")
plt.savefig(rf'plots\22b_plot1.png')
plt.close()

print("Plot 1 has been saved to the 'plots' folder.")

print("\n###########################################################\n")

#################################################################################

print("Problem 2.2.c\n")
print("Calculating Approximate Generalized Squared Error for Single Trees:")

num_reps = 100

num_trees = 100

all_bagged_models = []

for i in range(num_reps):
    train_subdata = train_np[np.random.choice(5000, 1000, replace=False), :]
    
    bagged_trees = []
    
    for j in range(num_trees):
        
        
        bs_train = train_np[np.random.randint(0,1000,1000),:]
    
        bs_X = np.delete(bs_train, -1, axis=1)
        bs_Y = bs_train[:,-1].tolist()
        
        tree = dt1.DecisionTree()
        tree.root = dt1.ID3(bs_X, bs_Y, Attrs_bank)
        
        bagged_trees.append(tree)
    
    all_bagged_models.append(bagged_trees)
        
single_bias_terms = []
single_var_terms = []

for index in range(len(test_X_np)):
    
    row_pred = 0
    predictions = []

    for i in all_bagged_models:
        
        current_pred = i[0].predict(test_X_np[index], Attrs_bank)
        
        predictions.append(current_pred)
        
        row_pred += current_pred / num_reps
        
    
    bias_term = (test_Y[index] - row_pred)**2
    
    var_term = 0
    
    for i in predictions:
        var_term += (i - row_pred)**2
        
    var_term /= (num_reps - 1)
    
    
    single_bias_terms.append(bias_term)
    single_var_terms.append(var_term)
    

single_bias = sum(single_bias_terms) / len(single_bias_terms)
single_var = sum(single_var_terms) / len(single_var_terms)

single_gen_sq_er = single_bias + single_var

print(round(single_gen_sq_er, 4))

print("\nCalculating Approximate Generalized Squared Error for Bagged Trees:")

bagged_bias_terms = []
bagged_var_terms = []

for index in range(len(test_X_np)):
    
    row_pred = 0
    predictions = []

    for i in all_bagged_models:
        
        current_pred = 0
        
        for j in range(len(i)):
            
            current_pred += i[j].predict(test_X_np[index], Attrs_bank)
            
            if j > int((len(i) - 1) / 2):
                if abs(current_pred) > len(i) - 1 - j:
                    break
            
        if current_pred > 0:
            current_pred = 1
        else:
            current_pred = -1
        
        predictions.append(current_pred)
        
        row_pred += current_pred / num_reps
        
    
    bias_term = (test_Y[index] - row_pred)**2
    
    var_term = 0
    
    for i in predictions:
        var_term += (i - row_pred)**2
        
    var_term /= (num_reps - 1)
    
    
    bagged_bias_terms.append(bias_term)
    bagged_var_terms.append(var_term)


bagged_bias = sum(bagged_bias_terms) / len(bagged_bias_terms)
bagged_var = sum(bagged_var_terms) / len(bagged_var_terms)

bagged_gen_sq_er = bagged_bias + bagged_var

print(round(bagged_gen_sq_er, 4))

print("\n###########################################################\n")
#################################################################################
print("Problem 2.2.d\n")
print("Creating plot 1: Train & Test Error for Random Forests (subset = 2)")

attr_subsets = [2, 4, 6]

random_forest = []

T = 500

for i in range(len(attr_subsets)):
    
    random_forest.append([])

    for j in range(T):
        
        bs_train = train_np[np.random.randint(0,5000,5000),:]
        
        bs_X = np.delete(bs_train, -1, axis=1)
        bs_Y = bs_train[:,-1].tolist()
        
        rand_tree = dt1.DecisionTree()
        rand_tree.root = dt1.ID3(bs_X, bs_Y, Attrs_bank, rand_forest=True, n_rand_attrs=attr_subsets[i])
        
        random_forest[i].append(rand_tree)

plot_data = []

for i in range(len(random_forest)):
    
    plot_data.append([])
    
    # Vary the number of trees used in Random Forest
    num_trees = [10, 25, 50, 100, 200, T]
    # num_trees = [10, 25, 50, T]

    # Create nested lists to hold test and train accuracy for each number of iterations/trees above
    rf_train_acc = []
    rf_test_acc = []

    for ele in num_trees:
        rf_train_acc.append([])
        rf_test_acc.append([])    
    
    # For each row in the training set
    for index in range(len(train_X_np)):
        
        # Set the current row prediction to zero
        row_pred = 0
        
        for j in range(len(random_forest[i])):

            row_pred += random_forest[i][j].predict(train_X_np[index], Attrs_bank)
            
            if j + 1 in num_trees:

                if row_pred > 0:
                    
                    rf_train_acc[num_trees.index(j+1)].append(1 == train_Y[index])
                    
                else:
                    
                    rf_train_acc[num_trees.index(j+1)].append(-1 == train_Y[index])
        
    
    # For each row in the testing set
    for index in range(len(test_X_np)):
        
        # Set the current row prediction to zero
        row_pred = 0
        
        for j in range(len(random_forest[i])):
            
            row_pred += random_forest[i][j].predict(test_X_np[index], Attrs_bank)
            
            if j + 1 in num_trees:
                
                if row_pred > 0:
                    
                    rf_test_acc[num_trees.index(j+1)].append(1 == test_Y[index])
                    
                else:
                    
                    rf_test_acc[num_trees.index(j+1)].append(-1 == test_Y[index])       
            
    # Create list and populate with training accuracies for plot
    Y1 = []
    for x in rf_train_acc:
        Y1.append(1 - (sum(x) / len(x)))
        
    # Create list and populate with testing accuracies for plot
    Y2 = []
    for x in rf_test_acc:
        Y2.append(1 - (sum(x) / len(x)))
    
    plot_data[i].append(Y1)
    plot_data[i].append(Y2)
    
# Populate X axis with each varied number of iterations/bagged trees
X = num_trees

train_2_Y = plot_data[0][0]
test_2_Y = plot_data[0][1]

train_4_Y = plot_data[1][0]
test_4_Y = plot_data[1][1]

train_6_Y = plot_data[2][0]
test_6_Y = plot_data[2][1]

# Create line plots for 2.2.d
plt.plot(X, train_2_Y, marker = 'o')
plt.plot(X, test_2_Y, marker = 'o')
plt.title("Train & Test Error for Random Forests")
plt.legend(['Train', "Test"])
plt.xlabel("# Iterations\n\nRandom Feature Subset Size 2")
plt.ylabel("Error")
plt.savefig(fr"plots\22d_plot1.png")
plt.close()

print("Plot 1 has been saved to the 'plots' folder.\n")

print("Creating plot 2: Train & Test Error for Random Forests (subset = 4)")


plt.plot(X, train_4_Y, marker = 'o')
plt.plot(X, test_4_Y, marker = 'o')
plt.title("Train & Test Error for Random Forests")
plt.legend(['Train', "Test"])
plt.xlabel("# Iterations\n\nRandom Feature Subset Size 4")
plt.ylabel("Error")
plt.savefig(fr"plots\22d_plot2.png")
plt.close()

print("Plot 2 has been saved to the 'plots' folder.\n")

print("Creating plot 3: Train & Test Error for Random Forests (subset = 6)")

plt.plot(X, train_6_Y, marker = 'o')
plt.plot(X, test_6_Y, marker = 'o')
plt.title("Train & Test Error for Random Forests")
plt.legend(['Train', "Test"])
plt.xlabel("# Iterations\n\nRandom Feature Subset Size 6")
plt.ylabel("Error")
plt.savefig(fr"plots\22d_plot3.png")
plt.close()

print("Plot 3 has been saved to the 'plots' folder.")

print("\n###########################################################\n")
#################################################################################
print("Problem 2.2.e\n")

print("Calculating Approximate Generalized Squared Error for Single Random Trees:")

num_reps = 100

num_trees = 100

all_forests = []

for i in range(num_reps):
    train_subdata = train_np[np.random.choice(5000, 1000, replace=False), :]
    
    random_trees = []
    
    for j in range(num_trees):
        
        bs_train = train_np[np.random.randint(0,1000,1000),:]
    
        bs_X = np.delete(bs_train, -1, axis=1)
        bs_Y = bs_train[:,-1].tolist()
        
        tree = dt1.DecisionTree()
        tree.root = dt1.ID3(bs_X, bs_Y, Attrs_bank, rand_forest=True, n_rand_attrs=5)
        
        random_trees.append(tree)
    
    all_forests.append(random_trees)

single_bias_terms = []
single_var_terms = []

for index in range(len(test_X_np)):
    
    row_pred = 0
    predictions = []

    for i in all_forests:
        
        current_pred = i[0].predict(test_X_np[index], Attrs_bank)
        
        predictions.append(current_pred)
        
        row_pred += current_pred / num_reps
        
    
    bias_term = (test_Y[index] - row_pred)**2
    
    var_term = 0
    
    for i in predictions:
        var_term += (i - row_pred)**2
        
    var_term /= (num_reps - 1)
    
    
    single_bias_terms.append(bias_term)
    single_var_terms.append(var_term)
    
single_bias = sum(single_bias_terms) / len(single_bias_terms)
single_var = sum(single_var_terms) / len(single_var_terms)

single_gen_sq_er = single_bias + single_var
        
print(round(single_gen_sq_er, 4))

print("\nCalculating Approximate Generalized Squared Error for Random Forests:")

forest_bias_terms = []
forest_var_terms = []

for index in range(len(test_X_np)):
    
    row_pred = 0
    predictions = []

    for i in all_forests:
        
        current_pred = 0
        
        for j in range(len(i)):
            
            current_pred += i[j].predict(test_X_np[index], Attrs_bank)
            
            if j > int((len(i) - 1) / 2):
                if abs(current_pred) > len(i) - 1 - j:
                    break
            
        if current_pred > 0:
            current_pred = 1
        else:
            current_pred = -1
        
        predictions.append(current_pred)
        
        row_pred += current_pred / num_reps
        
    bias_term = (test_Y[index] - row_pred)**2
    
    var_term = 0
    
    for i in predictions:
        var_term += (i - row_pred)**2
        
    var_term /= (num_reps - 1)
    
    forest_bias_terms.append(bias_term)
    forest_var_terms.append(var_term)

forest_bias = sum(forest_bias_terms) / len(forest_bias_terms)
forest_var = sum(forest_var_terms) / len(forest_var_terms)

forest_gen_sq_er = forest_bias + forest_var

print(round(forest_gen_sq_er, 4))


print("\n###########################################################\n")
#################################################################################

print("Reading & processing concrete data.")

col_names = ["Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr", "Fine Aggr", "SLUMP"]

concrete_train = pd.read_csv(fr"data\concrete-train.csv", names=col_names)
concrete_test = pd.read_csv(fr"data\concrete-test.csv", names=col_names)

concrete_train.insert(0, "Bias", np.full(len(concrete_train), 1))
concrete_test.insert(0, "Bias", np.full(len(concrete_test), 1))

conc_train = concrete_train.to_numpy()
conc_test = concrete_test.to_numpy()

conc_train_X = np.delete(conc_train, -1, axis=1)
conc_train_Y = conc_train[:,-1].tolist()

conc_test_X = np.delete(conc_test, -1, axis=1)
conc_test_Y = conc_test[:,-1].tolist()

print("Concrete data preparation complete.")

print("\n###########################################################\n")
#################################################################################

print("Problem 4.a\n")

print("Calculating learned weight vector using Batch Gradient Descent:")

def J(X, Y, weight):
    cost = 0
    for index in range(len(X)):
        cost += (Y[index] - np.dot(X[index], weight))**2
    return cost / 2

def J_gradient(X, Y, weight):
    gradient = []
    for col_index in range(len(X[0,:])):
        gradient_term = 0
        for row_index in range(len(X[:, 0])):
            gradient_term -= (Y[row_index] - np.dot(X[row_index,:], weight)) * X[row_index, col_index]
        gradient.append(gradient_term)
    return np.array(gradient)
        

def bgd(X, Y, r = 1, threshold = 10**-6):
    w = np.zeros(len(X[0,:]))
    cost = []
    norm = 1
    while norm > threshold:
        w_new = np.subtract(w, r * J_gradient(X, Y, w))
        norm = np.linalg.norm(w_new - w)
        w = w_new
        cost.append(J(X, Y, w))
    return w, cost

learning_rate = .014

linreg_model_bgd = bgd(conc_train_X, conc_train_Y, learning_rate)

print("Weight Vector:")
print(np.round(linreg_model_bgd[0], 4))
print("\nLearning Rate:")
print(learning_rate)
print("\nTest Set Cost:")
print(round(J(conc_test_X, conc_test_Y, linreg_model_bgd[0]), 4))

print("\nCreating plot 1: Batch Gradient Descent Cost Across Iterations")

plt.plot(range(len(linreg_model_bgd[1])), linreg_model_bgd[1])
plt.title("Batch Gradient Descent Cost Across Iterations")
plt.xlabel("Number Iterations/Updates")
plt.ylabel("Cost")
plt.savefig(rf'plots\4a_plot1.png')
plt.close()

print("Plot 1 has been saved to the 'plots' folder.")

print("\n###########################################################\n")
#################################################################################

print("Problem 4.b\n")

print("Calculating learned weight vector using Stochastic Gradient Descent:")

def J_gradient_stoch(X, Y, weight):
    gradient = []
    for index in range(len(X)):
        gradient.append((Y - np.dot(weight, X)) * X[index])
    return np.array(gradient)

def sgd(X, Y, r = 1, threshold = 10**-6):
    w = np.zeros(len(X[0,:]))
    cost = []
    norm = 1
    while norm > threshold:
        rand_index = np.random.randint(0, len(X))
        w_new = np.add(w, r * J_gradient_stoch(X[rand_index,:], Y[rand_index], w))
        norm = np.linalg.norm(w_new - w)
        w = w_new
        cost.append(J(X, Y, w))
    return w, cost

learning_rate = .002

linreg_model_sgd = sgd(conc_train_X, conc_train_Y, learning_rate)

print("Weight Vector:")
print(np.round(linreg_model_sgd[0], 4))
print("\nLearning Rate:")
print(learning_rate)
print("\nTest Set Cost:")
print(round(J(conc_test_X, conc_test_Y, linreg_model_sgd[0]), 4))

print("\nCreating plot 1: Stochastic Gradient Descent Cost Across Iterations")

plt.plot(range(len(linreg_model_sgd[1])), linreg_model_sgd[1])
plt.title("Stochastic Gradient Descent Cost Across Iterations")
plt.xlabel("Number Iterations/Updates")
plt.ylabel("Cost")
plt.savefig(rf'plots\4b_plot1.png')
plt.close()

print("Plot 1 has been saved to the 'plots' folder.\n")

print("Please note that due to the random selection of training examples, the output for problem 4.b may not match that presented in the write-up.")

print("\n###########################################################\n")
#################################################################################

print("Problem 4.c\n")

print("Calculating the weight vector using analytical method:")

print(np.round(np.matmul(np.linalg.inv(np.matmul(np.transpose(conc_train_X), conc_train_X)), np.matmul(np.transpose(conc_train_X), np.array(conc_train_Y))), 4))

print("\n###########################################################\n")
#################################################################################
print("Additional output used in 4.c explanation:\n")

print("Calculating average test cost from 20 iterations of Stochastic Gradient Descent:")

test_costs = []

for i in range(20):
    model_sgd = sgd(conc_train_X, conc_train_Y, learning_rate)
    
    test_costs.append(J(conc_test_X, conc_test_Y, model_sgd[0]))

print(round(np.mean(test_costs), 4))

print("\n###########################################################")
#################################################################################