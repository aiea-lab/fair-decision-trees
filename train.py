import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
import disc_func
from decision_tree import Node
from math import log, e

# def entropy(s):
#     '''s
#     Helper function, calculates entropy from an array of integer values.
    
#     :param s: list
#     :return: float, entropy value
#     '''
#     # Convert to integers to avoid runtime errors
#     counts = np.bincount(np.array(s, dtype=np.int64))
#     # Probabilities of each class label
#     percentages = counts / len(s)

#     # Caclulate entropy
#     entropy = 0
#     for pct in percentages:
#         if pct > 0:
#             entropy += pct * np.log2(pct)
#     return -entropy

def entropy(labels, base=2):
  """ Computes entropy of label distribution. """

  n_labels = len(labels)

  if n_labels <= 1:
    return 0

  value,counts = np.unique(labels, return_counts=True)
  probs = counts / n_labels
  n_classes = np.count_nonzero(probs)

  if n_classes <= 1:
    return 0

  ent = 0.

  # Compute entropy
  base = e if base is None else base
  for i in probs:
    ent -= i * log(i, base)

  return ent


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


def statParity_equalOdds(data):
    '''
    data: 3d array-like
    protectedIndex: int index of the protected class
    return average of statistical parity and Equalized odds
    '''
    if data.size == 0 or np.sum(data[0,:,:]) == 0 or np.sum(data[1,:,:])==0 or np.sum(data[1,1,:]) == 0 or np.sum(data[0,1,:]) == 0:
        return -100
    spd = np.sum(data[0,:,1])/np.sum(data[0,:,:])-np.sum(data[1,:,1])/np.sum(data[1,:,:])
    eod = np.sum(data[0,1,1])/np.sum(data[0,1,:])-np.sum(data[1,1,1])/np.sum(data[1,1,:])
    return (spd+eod)/2


def gain_root(df_left, df_right, disc_index, disc_data):
    # print(type(root_data))
    pred_left = round(np.mean(df_left[:,-2]))
    pred_right = round(np.mean(df_right[:,-2]))
    disc_old = statParity_equalOdds(disc_data)
    if disc_data.size == 0 or np.sum(disc_data[0,:,:]) == 0 or np.sum(disc_data[1,:,:])==0 or np.sum(disc_data[1,1,:]) == 0 or np.sum(disc_data[0,1,:]) == 0:
        return -100, disc_data
    acc_old = np.sum(disc_data[:,:,:])/(np.sum(disc_data[:,0,0])+np.sum(disc_data[:,1,1]))
    for row in df_left:
        disc_data[int(row[disc_index]), int(row[-2]), int(row[-1])] -= 1
        disc_data[int(row[disc_index]), int(row[-2]), pred_left] += 1
    for row in df_right:
        disc_data[int(row[disc_index]), int(row[-2]), int(row[-1])] -= 1
        disc_data[int(row[disc_index]), int(row[-2]), pred_right] += 1
    disc_new = statParity_equalOdds(disc_data)
    if disc_data.size == 0 or np.sum(disc_data[0,:,:]) == 0 or np.sum(disc_data[1,:,:])==0 or np.sum(disc_data[1,1,:]) == 0 or np.sum(disc_data[0,1,:]) == 0:
        return -100, disc_data
    acc_new = np.sum(disc_data[:,:,:])/(np.sum(disc_data[:,0,0])+np.sum(disc_data[:,1,1]))
    disc_delta = disc_new-disc_old
    acc_delta = acc_new-acc_old

    y = np.append(df_left, df_right, axis=0)[:, -2]
    y_left = df_left[:, -2]
    y_right = df_right[:, -2]
    # Caclulate the information gain and save the split parameters
    # if the current split if better then the previous best
    info_gain = information_gain(y, y_left, y_right)
    # if disc_delta < -0.01:
    #     return -100, disc_data
    return disc_delta, disc_data
    # return 4*disc_delta+min(acc_delta, info_gain), disc_data

def gain_calc(df_left, df_right, protected_index, disc_function=disc_func.discrimination):
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
    discrimination = disc_function(data, protected_index)
    gain = (info_gain + discrimination + 1)/2
    return gain

def gain_disc_priority(df_left, df_right, protected_index, disc_function=disc_func.discrimination):
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
    discrimination = disc_function(data, protected_index)
    gain = (info_gain + 3*discrimination + 2)/4
    return gain

def gain_info_priority(df_left, df_right, protected_index, disc_function=disc_func.discrimination):
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
    discrimination = disc_function(data, protected_index)
    gain = (3*info_gain + discrimination + 2)/4
    return gain


class DecisionTreeFair:
    '''
    Class which implements a decision tree classifier algorithm.
    '''
    def __init__(self, disc_data, min_samples_split=2, max_depth=5, train_method=None, disc_index=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self.train_method = train_method
        self.disc_index = disc_index
        self.disc_data = disc_data
        # print('type', self.root_data)
    
    def _best_split(self, df):
        '''
        Helper function, calculates the best split for given features and target
        
        :param X: np.array, features
        :param y: np.array or list, target
        :return: dict
        '''
        best_split = {}
        best_gain = -9999
        best_disc_data = self.disc_data
        n_rows, n_cols = df.shape
        
        # For every dataset feature
        print('split', n_rows, np.unique(df[:,:-2]).shape)
        for f_idx in range(n_cols-2):
            # X_curr = df[:, f_idx]
            # print('cur idx', f_idx)
            unique = np.unique(df[:,f_idx])
            unique = np.sort(unique)[1::max(len(unique)//100,1)]
            # For every unique value of that feature
            for threshold in np.unique(unique):
                # Construct a dataset and split it to the left and right parts
                # Left part includes records lower or equal to the threshold
                # Right part includes records higher than the threshold
                # print(X.shape, y.reshape(1, -1).T.shape)
                df_left = df[df[:,f_idx] <= threshold]
                df_right = df[df[:,f_idx] > threshold]


                # Do the calculation only if there's data in both subsets
                if len(df_left) > 0 and len(df_right) > 0:
                    gain, disc_data = gain_root(df_left, df_right, self.disc_index, np.copy(self.disc_data))
                    if gain > best_gain:
                        best_split = {
                            'feature_index': f_idx,
                            'threshold': threshold,
                            'df_left': df_left,
                            'df_right': df_right,
                            'gain': gain
                        }
                        best_gain = gain
                        best_disc_data = disc_data
        if best_gain <= -10:
            print('worst case senario')
            print(best_split)
        self.disc_data = best_disc_data
        return best_split
    
    def _build(self, df, depth=0):
        '''
        Helper recursive function, used to build a decision tree from the input data.
        
        :param X: np.array, features
        :param y: np.array or list, target
        :param depth: current depth of a tree, used as a stopping criteria
        :return: Node
        '''
        n_rows, n_cols = df.shape
        
        # Check to see if a node should be leaf node
        if n_rows >= self.min_samples_split and depth <= self.max_depth:
            # Get the best split
            best = self._best_split(df)
            # If the split isn't pure
            if best['gain'] > 0:
                # Build a tree on the left
                left = self._build(
                    df=best['df_left'],
                    depth=depth + 1
                )
                right = self._build(
                    df=best['df_right'],
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
            leaf=round(np.mean(df[:,-2]))
        )
    
    def fit(self, df):
        '''
        Function used to train a decision tree classifier model.
        
        :param X: np.array, features
        :param y: np.array or list, target
        :return: None
        '''
        # Call a recursive function to build the tree
        pred = round(np.mean(df[:,-2]))
        for row in df:
            self.disc_data[int(row[self.disc_index]), int(row[-2]), int(row[-1])] -= 1
            self.disc_data[int(row[self.disc_index]), int(row[-2]), pred] += 1
        self.root = self._build(df)
        
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