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
        


# Entropy helper function with weights parameter to account for fractional examples
def H(data, weights):
    entropy = 0    
    weights_sum = sum(weights)
    
    weighted_proportions = dict()
    
    for i in data.unique().tolist():
        weighted_proportions.update({i : 0})
        for j, k in zip(data, weights):
            if i == j:
                weighted_proportions[i] += k
        weighted_proportions[i] /= weights_sum
        entropy -= weighted_proportions[i]*np.log2(weighted_proportions[i])
        
    return entropy
    
# Majority error helper function with weights parameter to account for fractional examples
def ME(data, weights):
    high_prop = 0
    weights_sum = sum(weights)
    weighted_proportions = dict()
    
    for i in data.unique().tolist():
        weighted_proportions.update({i : 0})
        for j, k in zip(data, weights):
            if i == j:
                weighted_proportions[i] += k
        weighted_proportions[i] /= weights_sum
    
        if weighted_proportions[i] > high_prop:
            high_prop = weighted_proportions[i]

    return 1 - high_prop

# Gini index helper function with weights parameter to account for fractional examples
def GI(data, weights):
    gini_index = 1.0
    weights_sum = sum(weights)
    weighted_proportions = dict()
    
    for i in data.unique().tolist():
        weighted_proportions.update({i : 0})
        for j, k in zip(data, weights):
            if i == j:
                weighted_proportions[i] += k
        weighted_proportions[i] /= weights_sum
        gini_index -= weighted_proportions[i]**2
        
    return gini_index

# Heuristic function that uses the specified heuristic helper function based on heur input parameter
def heuristic(data, heur, weights):
    if heur == "H":
        return H(data, weights)
    elif heur == "ME":
        return ME(data, weights)
    elif heur == "GI":
        return GI(data, weights)
    else:
        print("Invalid heuristic, options are:\nH - Entropy\nME - Majority Error\nGI - Gini Index")
        return None

# Function that takes the label column and weights as input parameters and returns the name and proportion of the most common label
def common_label(data, weights):
    weights_sum = sum(weights)
    weighted_counts = dict()
    
    for i in data.unique().tolist():
        weighted_counts.update({i : 0})
        for j, k in zip(data, weights):
            if i == j:
                weighted_counts[i] += k
        weighted_counts[i] /= weights_sum
    
    label_proportion = 0
    label_name = None
    
    for i in weighted_counts:
        if weighted_counts[i] > label_proportion:
            label_proportion = weighted_counts[i]
            label_name = i

    return label_name, label_proportion


# Function used to calculate information gain of one attribute using the specified heuristic
def IG(data, attr_name, attr_vals, label_var, heur, weights):
    exp_decrease = 0
    weights_sum = sum(weights)
    
    for x in attr_vals:
        Sv = data[data[attr_name] == x]
        if len(Sv) != 0:
            Sv_weights = [weights[i] for i in list(Sv.index)]
            Sv_weights_sum = sum(Sv_weights)
            exp_decrease += (Sv_weights_sum / weights_sum) * heuristic(Sv[label_var], heur, Sv_weights)
    return heuristic(data[label_var], heur, weights) - exp_decrease

# Function used to determine the best attribute to split the data on using the specified attributes and heuristic
def best_attr(data, a_dict, label_var, heur, weights, descriptive_output = True):   
    max_gain = -1
    best_attr_name = None
    for a_name in a_dict:
        temp_gain = IG(data, a_name, a_dict[a_name], label_var, heur, weights)
        if descriptive_output:
            print(f"\tIG({a_name}) = {round(temp_gain, 4)}")
        if max_gain < temp_gain:
            max_gain = temp_gain
            best_attr_name = a_name
    if descriptive_output:
        print()   
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
def ID3(S, L, attr_dict, parent = None, rule = None, heur = "H", max_depth = np.inf, weights = None, descriptive_output = True):
    # If weights are not given then default of 1 is ascribed to each row in the data
    if weights == None:
        weights = [1] * len(S)
    
    # Provides descriptive output for each step in tree construction if desired
    if descriptive_output:
        if parent == None:
            print("Full dataset:")
            print(f"{S}\n")
        else:
            print(f"Data subset where {parent.attribute} = {rule}")
            print(f"{S}\n")
    
    # Determines whether the max_depth specified in the function call is appropriate
    if max_depth < 0:
        print("The value passed for the max_depth parameter was not positive, so a full decision tree will be created.")
        max_depth = np.inf
    
    # Sets the current depth, when parent = None then it is the first function call and depth is zero
    if parent == None:
        current_depth = 0
    else:
        current_depth = parent.depth + 1
    
    # Determines the most common label and its proportion factoring weights for fractional examples
    common_tuple = common_label(S[L], weights)
    
    # Generic ID3 exit condition, which also includes a condition for the max depth of the tree
    if common_tuple[1] == 1.0 or len(attr_dict) == 0 or current_depth == max_depth:

        # Initialize leaf node and set the appropriate data members
        leaf = Node()
        leaf.parent = parent
        leaf.label = common_tuple[0]
        leaf.depth = current_depth

        # Provides descriptive output for each step in tree construction if desired
        if descriptive_output:
            if common_tuple[1] == 1.0:
                print(f"A leaf node was created (ID = {leaf.id}) because the highest label proportion is 100%")
            elif len(attr_dict) == 0:
                print(f"A leaf node was created (label = {common_tuple[0]}) because there are no remaining attributes")
            elif current_depth == max_depth:
                print(f"A leaf node was created (label = {common_tuple[0]}) because the maximum tree depth was reached")
        
        # Return leaf node
        return leaf
        
    else:
        
        # Initialize current root/non-leaf node and set the appropriate data members
        root = Node()
        root.parent = parent
        root.depth = current_depth
        
        # Set outheur based on the specified heuristic to be used in the descriptive output
        if heur == "H":
            outheur = "entropy"
        elif heur == "ME":
            outheur = "majority error"
        elif heur == "GI":
            outheur = "gini index"
        
        # Provides descriptive output for each step in tree construction if desired
        if descriptive_output:
            print(f"Information gain was calculated using {outheur} for the following remaining attributes: {list(attr_dict.keys())}")
        
        # Determine best attribute to split on, and pass it to the current node's attribute data member
        root.attribute = best_attr(S, attr_dict, L, heur, weights, descriptive_output=descriptive_output)[0]
        
        # Provides descriptive output for each step in tree construction if desired
        if descriptive_output:
            print(f"The best attribute to split the data is {root.attribute}")
        
        # Create new attribute dictionary used in the recursive call
        new_attr_dict = attr_dict.copy()
        
        # Pull the list of possible attribute values from the best-split attribute
        attr_values = new_attr_dict[root.attribute]
        
        # Remove best-split attribute from new attribute dictionary
        new_attr_dict.pop(root.attribute)
        
        # Provides descriptive output for each step in tree construction if desired
        if descriptive_output:
            print(f"The dataset will be split into {len(attr_values)} subsets based on the following conditions:")
            for i in attr_values:
                print(f"\t{root.attribute} = {i}")
            
            print()
        
        # For each possible value, v, of the best-split attribute:
        for v in attr_values:
            # Create data subset where best-split attribute equals v, and removes that attribute from the data subset
            Sv = S[S[root.attribute] == v].drop(root.attribute, axis=1)
            
            # Selects the weights associated with each example in the data subset to be used in the recursive call
            Sv_weights = [weights[i] for i in Sv.index.tolist()]
            
            # Reset the index of the data subset to start from 0
            Sv.reset_index(inplace=True, drop=True)
            
            # If the data subset is empty:
            if len(Sv) == 0:
                
                # Initialize leaf node and set appropriate data members
                leaf = Node()
                leaf.parent = root
                leaf.label = common_label(S[L], weights)[0]
                leaf.depth = root.depth + 1
                
                # Provides descriptive output for each step in tree construction if desired
                if descriptive_output:
                    print(f"A leaf node was created (ID = {leaf.id}) because the current data subset is empty.")
            
            else:
                
                # Initialize leaf node and set as the output from the next recursive call    
                leaf = ID3(Sv, L, new_attr_dict, root, v, heur, max_depth, Sv_weights, descriptive_output=descriptive_output)
            
            # Set the rule data member for the current leaf node
            leaf.rule = v
            
            # Append leaf node to the list of children nodes for the root/current node
            root.children.append(leaf)
        
        # Return the current node
        return root
    
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
    def predict(self, data):
        current = self.root
        while current.label == None:
            for x in current.children:
                if data[current.attribute] == x.rule:
                    current = x
                    break
        return current.label