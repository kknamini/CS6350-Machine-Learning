import numpy as np

# Decision Tree Node Class Definition
class _Node:
    # Class varaible used to generate ID values for nodes automatically upon initialization
    static_node_id_generator = 1
    
    # Initialization for Decision Tree Node
    def __init__(self):
        self.id = _Node.static_node_id_generator
        # Increment node generator variable
        _Node.static_node_id_generator += 1
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
    def _reset_id_gen():
        _Node.static_node_id_generator = 1
        
        
    # Helper function used to display the links between all nodes of a subtree
    def _links(node):        
        if len(node.children) > 0:
            for n in node.children:
                print(f"{node.id} -> {n.id}")
                _Node._links(n)
        else:
            print(f"Node {node.id} is a leaf")
        
    # Helper function used to display the details of all nodes of a subtree
    def _node_details(node):
        print(node)
        if len(node.children) > 0:
            for n in node.children:
                _Node._node_details(n)


# Entropy function w/ ability to handle fractional/weighted examples
def entropy(Y, weights):
    h = 0
    weights_sum = sum(weights)
    
    weighted_proportions = dict()
    
    for i in set(Y):
        weighted_proportions.update({i : 0})
        for j, k in zip(Y, weights):
            if i == j:
                weighted_proportions[i] += k
        weighted_proportions[i] /= weights_sum
        h -= weighted_proportions[i]*np.log2(weighted_proportions[i])
        
    return h

# Majority error function w/ ability to handle fractional/weighted examples
def majority_error(Y, weights):
    me = 0
    weights_sum = sum(weights)
    weighted_proportions = dict()
    
    for i in set(Y):
        weighted_proportions.update({i : 0})
        for j, k in zip(Y, weights):
            if i == j:
                weighted_proportions[i] += k
        weighted_proportions[i] /= weights_sum
    
        if weighted_proportions[i] > me:
            me = weighted_proportions[i]

    return 1 - me

# Gini index function w/ ability to handle fractional/weighted examples
def gini(Y, weights):
    gi = 1
    weights_sum = sum(weights)
    weighted_proportions = dict()
    
    for i in set(Y):
        weighted_proportions.update({i : 0})
        for j, k in zip(Y, weights):
            if i == j:
                weighted_proportions[i] += k
        weighted_proportions[i] /= weights_sum
        gi - weighted_proportions[i]**2
        
    return gi

# Calculates the appropriate metric for best split based on classifier initialization
def metric(Y, weights, split_metric):
    if split_metric == "entropy":
        return entropy(Y, weights)
    elif split_metric == "majority_error":
        return majority_error(Y, weights)
    elif split_metric == "gini":
        return gini(Y, weights)
    else:
        print("Invalid split metric, options are:\n'entropy'\n'majority_error'\n'gini'")
        return None


# Calculates the most common label, returns label value and proportion, can handle fractional/weighted examples
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


# Function used to calculate information gain of one attribute using the specified split metric
def gain(X, Y, col_id, attr_vals, weights, split_metric):
    exp_decrease = 0
    
    for x in attr_vals:

        row_indeces = np.where(X[:, col_id] == x)
        
        Sv_X = X[row_indeces]
        
        if len(Sv_X) != 0:
            
            Sv_weights = [weights[i] for i in row_indeces[0].tolist()]
            Sv_Y = [Y[i] for i in row_indeces[0].tolist()]
            exp_decrease += (sum(Sv_weights) / sum(weights)) * metric(Sv_Y, Sv_weights, split_metric)
    return metric(Y, weights, split_metric) - exp_decrease

# Function used to determine the best attribute to split the data on using the specified attributes
def best_split(X, Y, a_dict, weights, split_metric):   
    max_gain = -1
    best_attr_name = None
    for a_name in a_dict:
        
        col_id = 0
        
        for i in a_dict:
            if i != a_name:
                col_id += 1
            else:
                break
        
        
        temp_gain = gain(X, Y, col_id, a_dict[a_name], weights, split_metric)

        if max_gain < temp_gain:
            max_gain = temp_gain
            best_attr_name = a_name

    return best_attr_name, max_gain


# ID3 Algorithm to be used in tree construction
def ID3(X, Y, attr_dict, weights, split_metric, max_depth = np.inf, rand_tree = False, n_rand_attrs = None, parent = None, rule = None):
    
    
    if weights == None:
        weights = [1] * len(X)
    
    
    if max_depth < 0:
        print("The value passed for the max_depth parameter was negative, so a full decision tree will be created.")
        max_depth = np.inf
    # elif max_depth > len(attr_dict) - 1:
    #     max_depth = len(attr_dict) - 1
    
    if parent == None:
        current_depth = 0
    else:
        current_depth = parent.depth + 1
    
    common_tuple = common_label(Y, weights)
    
    if common_tuple[1] == 1.0 or len(attr_dict) == 0 or current_depth == max_depth:
        
        leaf = _Node()
        leaf.parent = parent
        leaf.label = common_tuple[0]
        leaf.depth = current_depth
        
        return leaf
        
    else:
        
        root = _Node()
        root.parent = parent
        root.depth = current_depth
        
        if rand_tree:
            
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
                
            root.attribute = best_split(X, Y, rand_dict, weights, split_metric)[0]
            
        else:
            root.attribute = best_split(X, Y, attr_dict, weights, split_metric)[0]
        
        
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
                
                leaf = _Node()
                leaf.parent = root
                leaf.label = common_label(Y, weights)[0]
                leaf.depth = root.depth + 1
            
            else:
                
                leaf = ID3(Sv_X, Sv_Y, new_attr_dict, Sv_weights, split_metric, max_depth, rand_tree, n_rand_attrs, root, v)
            
            leaf.rule = v
            
            root.children.append(leaf)
            
        return root


class DecisionTree:
    
    # Initialization for decision tree
    def __init__(self, split_metric = "gini", max_depth = np.inf, rand_tree = False, n_rand_attrs = None):
        self.root = None
        self.split_metric = split_metric
        self.max_depth = max_depth
        self.rand_tree = rand_tree
        self.n_rand_attrs = n_rand_attrs
        
    # Display the details of all nodes
    def display_tree(self):
        _Node._node_details(self.root)
    
    # Display the links between all nodes, and whether they are leaf nodes or not
    def display_links(self):
        _Node._links(self.root)

    # Function to fit data to tree structure using ID3 algorithm. Automatically resets Node ID numbers after tree construction
    def fit(self, X, y, attr_dict, weights = None):
        self.root = ID3(X, y, attr_dict, weights, self.split_metric, self.max_depth, self.rand_tree, self.n_rand_attrs)
        _Node._reset_id_gen()
    
    # Make one prediction      
    def _predict_one(self, one_row, a_dict):
        current = self.root
        while current.label == None:
            for x in current.children:
                
                index = 0
                
                for i in a_dict:
                    if i != current.attribute:
                        index += 1
                    else:
                        break
                
                
                if one_row[index] == x.rule:
                    current = x
                    break
        return current.label
    
    # Create predictions on full dataset
    def predict(self, data, a_dict):
        preds = []
        
        for row in data:
            preds.append(self._predict_one(row, a_dict))
            
        preds = np.array(preds)
        
        return preds