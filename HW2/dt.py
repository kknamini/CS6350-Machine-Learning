import numpy as np
import pandas as pd

# Used to suppress false-positive warning message
pd.options.mode.chained_assignment = None

# Decision Tree Node Class Definition
class Node:
    # Class varaible used to generate ID values for nodes automatically upon initialization
    static_node_id_generator = 1
    
    # Initialization for Decision Tree Node
    def __init__(self):
        self.id = Node.static_node_id_generator
        # Increment node generator variable
        Node.static_node_id_generator += 1
        self.parent = None
        self.children = []
        self.attribute = None
        self.rule = None
        self.label = None
        self.depth = 0
        
    # String representation for Decision Tree Node
    def __str__(self):
        outstr = f"""Node Details:
ID: {self.id}
Depth: {self.depth}
Attribute: {self.attribute}
Rule: {self.rule}
label: {self.label}
"""
        return outstr
    
    # Node member function used to reset id generator variable
    # This should be called after a Decision Tree is created if you want subsequent decision tree node id's to begin with 1
    def reset_id_gen():
        Node.static_node_id_generator = 1
        
# Helper function used to display the links between all nodes of a subtree
def links(node):        
    if len(node.children) > 0:
        for n in node.children:
            print(f"{node.id} -> {n.id}")
            links(n)
    else:
        print(f"Node {node.id} is a leaf")
    
# Helper function used to display the details of all nodes of a subtree
def node_details(node):
    print(node)
    if len(node.children) > 0:
        for n in node.children:
            node_details(n)

# Entropy helper function with weights parameter to account for fractional examples
def H(Y, weights):
    entropy = 0    
    
    
    weights_sum = sum(weights)
    
    weighted_proportions = dict()
    
    for i in set(Y):
        weighted_proportions.update({i : 0})
        for j, k in zip(Y, weights):
            if i == j:
                weighted_proportions[i] += k
        weighted_proportions[i] /= weights_sum
        entropy -= weighted_proportions[i]*np.log2(weighted_proportions[i])
    
    # numpy_Y = np.array(Y)
    # w = np.array(weights)
        
    # unq,inv=np.unique(numpy_Y,return_inverse=True)
    # freqs = np.bincount(inv,w.reshape(-1)) / sum(w)

    # for i in range(len(unq)):
    #     entropy -= freqs[i]*np.log2(freqs[i])
        
    return entropy
    
# Majority error helper function with weights parameter to account for fractional examples
def ME(Y, weights):
    high_prop = 0
    weights_sum = sum(weights)
    weighted_proportions = dict()
    
    for i in set(Y):
        weighted_proportions.update({i : 0})
        for j, k in zip(Y, weights):
            if i == j:
                weighted_proportions[i] += k
        weighted_proportions[i] /= weights_sum
    
        if weighted_proportions[i] > high_prop:
            high_prop = weighted_proportions[i]

    return 1 - high_prop

# Gini index helper function with weights parameter to account for fractional examples
def GI(Y, weights):
    gini_index = 1
    weights_sum = sum(weights)
    weighted_proportions = dict()
    
    for i in set(Y):
        weighted_proportions.update({i : 0})
        for j, k in zip(Y, weights):
            if i == j:
                weighted_proportions[i] += k
        weighted_proportions[i] /= weights_sum
        gini_index - weighted_proportions[i]**2
        
    return gini_index

# Heuristic function that uses the specified heuristic helper function based on heur input parameter
def heuristic(Y, heur, weights):
    if heur == "H":
        return H(Y, weights)
    elif heur == "ME":
        return ME(Y, weights)
    elif heur == "GI":
        return GI(Y, weights)
    else:
        print("Invalid heuristic, options are:\nH - Entropy\nME - Majority Error\nGI - Gini Index")
        return None

# Function that takes the label column and weights as input parameters and returns the name and proportion of the most common label
def common_label(Y, weights):
    
    weights_sum = sum(weights)
    weighted_counts = dict()
    
    label_proportion = 0
    label_name = None
    
    for i in set(Y):
        weighted_counts.update({i : 0})
        for j, k in zip(Y, weights):
            if i == j:
                weighted_counts[i] += k
        weighted_counts[i] /= weights_sum
    
        if weighted_counts[i] > label_proportion:
            label_proportion = weighted_counts[i]
            label_name = i

    return label_name, label_proportion

# Function used to calculate information gain of one attribute using the specified heuristic
def IG(X, Y, col_id, attr_vals, heur, weights):
    exp_decrease = 0
    
    for x in attr_vals:

        row_indeces = np.where(X[:, col_id] == x)
        
        Sv_X = X[row_indeces]
        
        if len(Sv_X) != 0:
            
            Sv_weights = [weights[i] for i in row_indeces[0].tolist()]
            Sv_Y = [Y[i] for i in row_indeces[0].tolist()]
            exp_decrease += (sum(Sv_weights) / sum(weights)) * heuristic(Sv_Y, heur, Sv_weights)
    return heuristic(Y, heur, weights) - exp_decrease

# Function used to determine the best attribute to split the data on using the specified attributes and heuristic
def best_attr(X, Y, a_dict, heur, weights):   
    max_gain = -1
    best_attr_name = None
    for a_name in a_dict:
        
        col_id = 0
        
        for i in a_dict:
            if i != a_name:
                col_id += 1
            else:
                break
        
        
        temp_gain = IG(X, Y, col_id, a_dict[a_name], heur, weights)
        # if descriptive_output:
        #     print(f"\tIG({a_name}) = {round(temp_gain, 4)}")
        if max_gain < temp_gain:
            max_gain = temp_gain
            best_attr_name = a_name
    # if descriptive_output:
    #     print()
    return best_attr_name, max_gain

# Function used to convert continuous data to binary data using the median value
# medians = None is used to convert the training data
# The altered dataset is returned, along with the calculated medians from the training data, which are then used to alter the testing dataset
def convert_numerical(data, data_types, medians = None):
    meds = []
    if medians == None:
        for i, j in zip(data_types, data.columns):
            if i == "num":
                med = np.median(data[j])
                meds.append(med)
                data[f"{j} > {med}"] = pd.cut(data[j], bins = [np.NINF, med, np.inf], labels=[0,1])
                data = data.drop(j, axis=1)
                
    else:
        median_index = 0
        for i, j in zip(data_types, data.columns):
            if i == "num":
                data[f"{j} > {medians[median_index]}"] = pd.cut(data[j], bins = [np.NINF, medians[median_index], np.inf], labels=[0,1])
                data = data.drop(j, axis=1)
                median_index += 1
    return data, meds
    

# ID3 algorithm with hyperparameters for the heuristic, max depth, and weights for fractional examples
# The following parameters are only meant for recursive calls within the ID3 code body: parent, rule
def ID3(X, Y, attr_dict, heur = "H", max_depth = np.inf, weights = None, parent = None, rule = None, rand_forest = False, n_rand_attrs = None):
    
    
    if weights == None:
        weights = [1] * len(X)
    
    # if descriptive_output:
    #     if parent == None:
    #         print("Full dataset:")
    #         print(f"{S}\n")
    #     else:
    #         print(f"Data subset where {parent.attribute} = {rule}")
    #         print(f"{S}\n")
    
    if max_depth < 0:
        print("The value passed for the max_depth parameter was not positive, so a full decision tree will be created.")
        max_depth = np.inf
    
    if parent == None:
        current_depth = 0
    else:
        current_depth = parent.depth + 1
    
    common_tuple = common_label(Y, weights)
    
    if common_tuple[1] == 1.0 or len(attr_dict) == 0 or current_depth == max_depth:
        
        leaf = Node()
        leaf.parent = parent
        leaf.label = common_tuple[0]
        leaf.depth = current_depth
        
        return leaf
        
    else:
        
        root = Node()
        root.parent = parent
        root.depth = current_depth
        
        if rand_forest:
            
            if n_rand_attrs == None:
                
                if len(attr_dict) > 2:
                    n_rand_attrs = int(len(attr_dict) / 2)
                else:
                    n_rand_attrs = 1
                    
            else:
                
                if n_rand_attrs > len(attr_dict):
                    n_rand_attrs = int(len(attr_dict) / 2)
                elif len(attr_dict) <= 2:
                    n_rand_attrs = 1
                    
            keys = np.array(list(attr_dict.keys()))
            keys = keys[np.random.choice(len(keys), n_rand_attrs, replace=False)].tolist()
            
            rand_dict = {}

            for i in keys:
                rand_dict[i] = attr_dict[i]
                
            root.attribute = best_attr(X, Y, rand_dict, heur, weights)[0]
            
        else:
            root.attribute = best_attr(X, Y, attr_dict, heur, weights)[0]
        
        
        col_id = 0
        
        for i in attr_dict:
            if i != root.attribute:
                col_id += 1
            else:
                break
        
        new_attr_dict = attr_dict.copy()
        attr_values = new_attr_dict[root.attribute]
        new_attr_dict.pop(root.attribute)
        
        for v in attr_values:
            
            row_indeces = np.where(X[:, col_id] == v)
            
            Sv_X = np.delete(X[row_indeces], col_id, 1)
            
            Sv_weights = [weights[i] for i in row_indeces[0].tolist()]
            Sv_Y = [Y[i] for i in row_indeces[0].tolist()]     
            
            if len(Sv_X) == 0:
                
                leaf = Node()
                leaf.parent = root
                leaf.label = common_label(Y, weights)[0]
                leaf.depth = root.depth + 1
            
            else:
                
                leaf = ID3(Sv_X, Sv_Y, new_attr_dict, heur, max_depth, Sv_weights, root, v, rand_forest, n_rand_attrs)
            
            leaf.rule = v
            
            root.children.append(leaf)
            
        return root

# Decision Tree Class Definition
class DecisionTree:
    
    # Initialization for decision tree
    def __init__(self):
        self.root = None
        
    # Member function to display the details of all nodes
    def display_tree(self):
        node_details(self.root)
    
    # Member function to display the links between all nodes, and whether they are leaf nodes or not
    def display_links(self):
        links(self.root)
    
    
    # Member function to make a prediction on one row of data        
    def predict(self, data, a_dict):
        current = self.root
        while current.label == None:
            for x in current.children:
                
                index = 0
                
                for i in a_dict:
                    if i != current.attribute:
                        index += 1
                    else:
                        break
                
                
                if data[index] == x.rule:
                    current = x
                    break
        return current.label   