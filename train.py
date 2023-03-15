import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
import disc_func
from decision_tree import Node


def entropy(s):
    '''
    Helper function, calculates entropy from an array of integer values.
    
    :param s: list
    :return: float, entropy value
    '''
    # Convert to integers to avoid runtime errors
    counts = np.bincount(np.array(s, dtype=np.int64))
    # Probabilities of each class label
    percentages = counts / len(s)

    # Caclulate entropy
    entropy = 0
    for pct in percentages:
        if pct > 0:
            entropy += pct * np.log2(pct)
    return -entropy


def information_gain(parent, left_child, right_child):
    '''
    Helper function, calculates information gain from a parent and two child nodes.
    
    :param parent: list, the parent node
    :param left_child: list, left child of a parent
    :param right_child: list, right child of a parent
    :return: float, information gain
    '''
    num_left = len(left_child) / len(parent)
    num_right = len(right_child) / len(parent)
    
    # One-liner which implements the previously discussed formula
    return entropy(parent) - (num_left * entropy(left_child) + num_right * entropy(right_child))


def gain_calc(df_left, df_right, protected_index):
    y = np.append(df_left, df_right, axis=0)[:, -1]
    y_left = df_left[:, -1]
    y_right = df_right[:, -1]
    pred_left = np.average(y_left) > 0.5
    pred_right = np.average(y_right) > 0.5
    # Caclulate the information gain and save the split parameters
    # if the current split if better then the previous best
    info_gain = information_gain(y, y_left, y_right)
    left_child = np.insert(df_left, -1, pred_left, axis=1)
    right_child = np.insert(df_right, -1, pred_right, axis=1)
    data = np.append(left_child, right_child, axis=0)
    discrimination = disc_func.statParity_equalOdds(data, protected_index)
    gain = (info_gain + discrimination + 1)/2
    return gain

def gain_disc_priority(df_left, df_right, protected_index):
    y = np.append(df_left, df_right, axis=0)[:, -1]
    y_left = df_left[:, -1]
    y_right = df_right[:, -1]
    pred_left = np.average(y_left) > 0.5
    pred_right = np.average(y_right) > 0.5
    # Caclulate the information gain and save the split parameters
    # if the current split if better then the previous best
    info_gain = information_gain(y, y_left, y_right)
    left_child = np.insert(df_left, -1, pred_left, axis=1)
    right_child = np.insert(df_right, -1, pred_right, axis=1)
    data = np.append(left_child, right_child, axis=0)
    discrimination = disc_func.statParity_equalOdds(data, protected_index)
    gain = (info_gain + 3*discrimination + 2)/4
    return gain

def gain_info_priority(df_left, df_right, protected_index):
    y = np.append(df_left, df_right, axis=0)[:, -1]
    y_left = df_left[:, -1]
    y_right = df_right[:, -1]
    pred_left = np.average(y_left) > 0.5
    pred_right = np.average(y_right) > 0.5
    # Caclulate the information gain and save the split parameters
    # if the current split if better then the previous best
    info_gain = information_gain(y, y_left, y_right)
    left_child = np.insert(df_left, -1, pred_left, axis=1)
    right_child = np.insert(df_right, -1, pred_right, axis=1)
    data = np.append(left_child, right_child, axis=0)
    discrimination = disc_func.statParity_equalOdds(data, protected_index)
    gain = (3*info_gain + discrimination + 2)/4
    return gain


class DecisionTreeFair:
    '''
    Class which implements a decision tree classifier algorithm.
    '''
    def __init__(self, min_samples_split=2, max_depth=5, train_method=None, protected_index=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self.train_method = train_method
        self.protected_index = protected_index
    
    def _best_split(self, X, y):
        '''
        Helper function, calculates the best split for given features and target
        
        :param X: np.array, features
        :param y: np.array or list, target
        :return: dict
        '''
        best_split = {}
        best_gain = -float('inf')
        n_rows, n_cols = X.shape
        
        # For every dataset feature
        for f_idx in range(n_cols):
            X_curr = X[:, f_idx]
            # For every unique value of that feature
            for threshold in np.unique(X_curr):
                # Construct a dataset and split it to the left and right parts
                # Left part includes records lower or equal to the threshold
                # Right part includes records higher than the threshold
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                df_left = np.array([row for row in df if row[f_idx] <= threshold])
                df_right = np.array([row for row in df if row[f_idx] > threshold])

                # Do the calculation only if there's data in both subsets
                if len(df_left) > 0 and len(df_right) > 0:
                    if self.train_method == None:
                        gain = gain_calc(df_left, df_right, self.protected_index)
                    else:
                        gain = self.train_method(df_left, df_right, self.protected_index)
                    if gain > best_gain:
                        best_split = {
                            'feature_index': f_idx,
                            'threshold': threshold,
                            'df_left': df_left,
                            'df_right': df_right,
                            'gain': gain
                        }
                        best_gain = gain
        return best_split
    
    def _build(self, X, y, depth=0):
        '''
        Helper recursive function, used to build a decision tree from the input data.
        
        :param X: np.array, features
        :param y: np.array or list, target
        :param depth: current depth of a tree, used as a stopping criteria
        :return: Node
        '''
        n_rows, n_cols = X.shape
        
        # Check to see if a node should be leaf node
        if n_rows >= self.min_samples_split and depth <= self.max_depth:
            # Get the best split
            best = self._best_split(X, y)
            # If the split isn't pure
            if best['gain'] > 0:
                # Build a tree on the left
                left = self._build(
                    X=best['df_left'][:, :-1], 
                    y=best['df_left'][:, -1],
                    depth=depth + 1
                )
                right = self._build(
                    X=best['df_right'][:, :-1], 
                    y=best['df_right'][:, -1],
                    depth=depth + 1
                )
                return Node(
                    feature=best['feature_index'], 
                    threshold=best['threshold'], 
                    less_node=left, 
                    greater_node=right, 
                    gain=best['gain']
                )
        # Leaf node - value is the most common target value 
        return Node(
            leaf=Counter(y).most_common(1)[0][0]
        )
    
    def fit(self, X, y):
        '''
        Function used to train a decision tree classifier model.
        
        :param X: np.array, features
        :param y: np.array or list, target
        :return: None
        '''
        # Call a recursive function to build the tree
        self.root = self._build(X, y)
        
    def _predict(self, x, tree):
        '''
        Helper recursive function, used to predict a single instance (tree traversal).
        
        :param x: single observation
        :param tree: built tree
        :return: float, predicted class
        '''
        # Leaf node
        if tree.leaf != None:
            return tree.leaf
        feature_value = x[tree.feature]
        
        # Go to the left
        if feature_value <= tree.threshold:
            return self._predict(x=x, tree=tree.less_node)
        
        # Go to the right
        if feature_value > tree.threshold:
            return self._predict(x=x, tree=tree.greater_node)
        
    def predict(self, X):
        '''
        Function used to classify new instances.
        
        :param X: np.array, features
        :return: np.array, predicted classes
        '''
        # Call the _predict() function for every observation
        return [self._predict(x, self.root) for x in X]

    def get_nodes(self):
        return self.root