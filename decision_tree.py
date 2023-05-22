import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import train
import disc_func
from sklearn import tree
import time

class Node:
    def __init__(self, leaf=None, feature=None, threshold=None, greater_node=None, less_node=None, data=None, gain=None, id=None):
        '''
        leaf: predicted class or None (if not leaf)
        feature: index of data node is comparing
        threshold: value of data node is comparing
        greater_node: greater node pointer
        less_node: less node pointer
        data: list of numpy arrays containing [attributes, true class, prediction]
        height: number of nodes above leaf
        gain: information/discrimination gain for training DT
        unique_id: random float for identifying parent/child relationship  
        '''
        self.leaf = leaf
        self.feature = feature
        self.threshold = threshold
        self.greater_node = greater_node
        self.less_node = less_node
        self.data = None
        if data == None:
            self.data = []
        else:
            self.data = data
        self.height = None
        self.gain = gain
        if id == None:
            self.unique_id = np.random.random((1))[0]
        else:
            self.unique_id = id
        self.disc_data = np.zeros(shape=(2,2,2)) # protected, class, prediction
    
    def add_greater_node(self, greater_node):
        self.greater_node = greater_node

    def add_less_node(self, less_node):
        self.less_node = less_node
    
    def is_leaf(self):
        return self.leaf != None
    
    def __eq__(self,obj):
        if obj == None:
            return False
        return self.unique_id == obj.unique_id

    def __repr__(self):
        if self.is_leaf():
            return "{class="+str(self.leaf)+"}"
        return "{"+str(self.feature)+"<="+str(self.threshold)+":"+str(self.less_node)+","+str(self.feature)+"<="+str(self.threshold)+":"+str(self.greater_node)+"}"
    
    def __str__(self):
        if self.is_leaf():
            return "{class="+str(self.leaf)+"}"
        return "{"+str(self.feature)+"<="+str(self.threshold)+":"+str(self.less_node)+","+str(self.feature)+"<="+str(self.threshold)+":"+str(self.greater_node)+"}"
    
    def __add__(self, x):
        if isinstance(x, Node):
            return x.unique_id+self.unique_id
        return x+self.unique_id
    
    def __radd__(self, x):
        return x+self.unique_id
    
    def export_json(self, attributes=None):
        if self.is_leaf():
            return {"class":self.leaf}
        if attributes == None:
            return {str(self.feature)+"<="+str(self.threshold):self.less_node.export_json(), str(self.feature)+">"+str(self.threshold):self.greater_node.export_json()}
        return {attributes[self.feature]+"<="+str(self.threshold):self.less_node.export_json(attributes), attributes[self.feature]+">"+str(self.threshold):self.greater_node.export_json(attributes)}

    def visual_json(self, disc_index, disc_func=disc_func.discrimination, attributes=None, highlight=[]):
        node_str = "{"
        if self in highlight:
            node_str += "\"Highlight\":\"TRUE\", "
        if self.is_leaf():
            node_str += "\"class\":"+str(self.leaf)+"}"
            return node_str
        if attributes == None:
            node_str += "\"discrimination\":\""+str(round(self.discrimination(disc_index, disc_func), 3))+"\", \"accuracy\":\""+str(round(self.accuracy(), 3))+"\",\"["+ str(self.feature)+"]\":\""+str(self.threshold)+"\", \"Left\":"+self.less_node.visual_json(disc_index, disc_func, attributes, highlight)+", \"Right\":"+self.greater_node.visual_json(disc_index, disc_func, attributes, highlight)+"}"
            return node_str
        return node_str + "\"discrimination\":\""+str(round(self.discrimination(disc_index, disc_func), 3))+"\", \"accuracy\":\""+str(round(self.accuracy(), 3))+"\",\""+attributes[self.feature]+"\":\""+str(self.threshold)+"\", \"Left\":"+self.less_node.visual_json(disc_index, disc_func, attributes, highlight)+", \"Right\":"+self.greater_node.visual_json(disc_index, disc_func, attributes, highlight)+"}"

    def add_data(self, data, disc_index):
        '''
        data: numpy array containing attributes and true class
        add_data: goes through the tree recursively appending data to each node with the predicted value
        return: predicted class
        '''
        if not self.is_leaf():
            if data[self.feature] <= self.threshold:
                leaf = self.less_node.add_data(data, disc_index)
            else:
                leaf = self.greater_node.add_data(data, disc_index)
            data = np.append(data, leaf)
            self.data.append(data)
            # print(data[disc_index], data[-2], leaf)
            self.disc_data[int(data[disc_index]), int(data[-2]), int(leaf)] += 1
            return leaf
        else:
            data = np.append(data, self.leaf)
            self.data.append(data)
            self.disc_data[int(data[disc_index]), int(data[-2]), int(self.leaf)] += 1
            return self.leaf
        
    def get_prediction(self, data):
        '''
        data: numpy array containing attributes (true class optional)
        return: predicted class
        '''
        if not self.is_leaf():
            if data[self.feature] <= self.threshold:
                return self.less_node.get_prediction(data)
            else:
                return self.greater_node.get_prediction(data)
        else:
            return self.leaf

    
    def get_leafs(self):
        '''
        return: list of leaf Nodes
        '''

        if self.is_leaf():
            return [self]
        leafs = []
        leafs.extend(self.less_node.get_leafs())
        leafs.extend(self.greater_node.get_leafs())
        return leafs

    def get_height(self, overwrite=False):
        '''
        overwrite: boolean on whether to recalculate height (if not already saved)
        return: height of node
        '''
        if self.height != None and overwrite==False:
            return self.height
        elif self.is_leaf():
            return 0
        else:
            self.height = max(self.less_node.get_height(overwrite), self.greater_node.get_height(overwrite)) + 1
            return self.height

    def get_layer(self, height):
        '''
        height: target node height
        return: list of nodes of given height
        '''
        if self.get_height() == height:
            return [self]
        elif self.get_height() < height:
            return []
        else:
            temp = self.less_node.get_layer(height)
            temp.extend(self.greater_node.get_layer(height))
            return temp

    def reset_data(self, height=False):
        '''
        height: boolean reset height
        reset_data: clean wipe of data
        '''
        if self.is_leaf():
            self.data = []
            if height:
                self.height = None
        else:
            self.data = []
            if height:
                self.height = None
            self.less_node.reset_data(height)
            self.greater_node.reset_data(height)

    def copy(self, copy_data=False):
        '''
        copy_data: boolean copy data
        return: recursive copy of node
        '''
        data = []
        if copy_data:
            for datum in self.data:
                data.append(np.copy(datum))
        if self.is_leaf():
            copy = Node(self.leaf, self.feature, self.threshold, data=data, id=self.unique_id)
            return copy
        else:
            copy = Node(self.leaf, self.feature, self.threshold, data=data, id=self.unique_id)
            if self.less_node != None:
                copy.add_greater_node(self.greater_node.copy(copy_data))
            if self.greater_node != None:
                copy.add_less_node(self.less_node.copy(copy_data))
            return copy

    def discrimination(self, disc_index, disc_func=disc_func.discrimination):
        '''
        disc_index: discriminatory index
        disc_fun: function to calculate discrimination
        return: discrimination
        '''
        return disc_func(self.data, disc_index)

    def accuracy(self, acc_func=disc_func.accuracy):
        '''
        acc_func: functin to calculate accuracy
        return: accuracy
        '''
        return acc_func(self.data)

    def retrain_node(self, disc_data, depth=None, train_method=None, disc_index=None, disc_func=disc_func.discrimination, criterion='gini', disc_coef=2, acc_coef=1):
        '''
        depth: max depth of new tree (None if use node's current height)
        stats: boolean prints statistics
        retrain_node: uses the data stored in the node to train a new tree
        train_method: function used instead of information gain for training
        disc_index: Index of discriminatory class
        return: new tree node
        '''
        if depth == None:
            depth = self.get_height()
        fair_model = train.DecisionTreeFair(disc_data, max_depth=depth-1,train_method=train_method, disc_index=disc_index, disc_coef=disc_coef, acc_coef=acc_coef)
        fair_model.fit(np.array(self.data))
        return fair_model.get_nodes()
    
    def get_parent(self, unique_id, height=None):
        '''
        unique_id: id of child node to match
        height: optional variable to help accelerate child identification
        return: parent node
        '''
        if self.is_leaf():
            return None
        elif height != None:
            potential_parents = self.get_layer(height)
            for parent in potential_parents:
                if parent.less_node.unique_id == unique_id or parent.greater_node.unique_id == unique_id:
                    return parent                
        elif self.less_node.unique_id == unique_id or self.greater_node.unique_id == unique_id:
            return self
        else:
            less_node = self.less_node.get_parent(unique_id)
            if less_node != None:
                return less_node
            greater_node = self.greater_node.get_parent(unique_id)
            if greater_node != None:
                return greater_node
        return None
    
    def replace_child(self, unique_id, node):
        '''
        unique_id: id of child
        node: node to replace child
        replace_child: swaps a current child with a new node
        '''
        if self.less_node.unique_id == unique_id:
            self.less_node = node
        elif self.greater_node.unique_id == unique_id:
            self.greater_node = node

    def simplify(self):
        '''
        simplify: if leafs are the same then become leaf
        '''
        if not self.is_leaf():
            self.less_node.simplify()
            self.greater_node.simplify()
            if self.less_node.leaf != None and self.less_node.leaf == self.greater_node.leaf:
                self.leaf = self.less_node.leaf
    
    def simplify_children(self, children, id=None):
        '''
        children: List of nodes
        return: simplified list of parents
        '''
        if self.is_leaf():
            return children
        children = self.less_node.simplify_children(children)
        children = self.greater_node.simplify_children(children)
        
        if id == None:
            id = []
            for child in children:
                id.append(child.unique_id)
        try:
            less_index = id.index(self.less_node.unique_id)
            greater_index = id.index(self.greater_node.unique_id)
            children.pop(max(less_index, greater_index))
            children.pop(min(less_index, greater_index))
            children.append(self)
        except ValueError:
            pass
        return children

def export_dict(clf):
    '''
    clf: sklearn decision tree
    return: node decision tree
    '''
    return export_dict_rec(tree.export_text(clf)[0:-1])
def export_dict_rec(tree_text):
    '''
    tree_text: string in the format of sklearn export text
    return: node decision tree
    '''
    if tree_text[0:12] == "|--- class: " or tree_text.find("feature") == -1:
        return Node(float(tree_text[12:]), None, None)
    tree_text = tree_text.replace("|--- feature_", "", 1)
    tree_text = tree_text.replace("\n|--- feature_", "\n", 1)
    tree_text = tree_text.replace("\n|   ", "\n")
    features = []
    feature = ""
    for letter in tree_text:
        if letter == " " or letter == "\n":
            features.append(feature)
            feature = ""
            if letter == "\n":
                break
        else:
            feature+=letter
    start = tree_text.find("\n")
    start_end = tree_text.find("\n"+tree_text[0])
    end = start_end+tree_text[start_end:].find("\n|")
    tree = Node(None, int(features[0]), float(features[-1]))
    tree.add_less_node(export_dict_rec(tree_text[start+1:start_end]))
    tree.add_greater_node(export_dict_rec(tree_text[end+1:]))
    return tree

def get_worst_node(dec_tree, target, disc_index, skip, disc_func=disc_func.discrimination):
    '''
    disc_index: index of discriminatory attribute
    disc_fun: function used for discrimination calculation
    get_worst_node: returns the node with worst discrimination
    return: node
    '''
    if dec_tree.is_leaf():
        return None
    worst_disc = dec_tree.discrimination(disc_index, disc_func)
    worst_node = dec_tree
    if dec_tree.unique_id in skip:
        # print('skip', dec_tree.unique_id, skip)
        worst_disc = 999
        worst_node = None
    if dec_tree.less_node != None:
        less_node = get_worst_node(dec_tree.less_node, target, disc_index, skip, disc_func)
        if less_node != None and worst_disc > less_node.discrimination(disc_index, disc_func):
            worst_node = less_node
            worst_disc = less_node.discrimination(disc_index, disc_func)
    if dec_tree.greater_node != None:
        greater_node = get_worst_node(dec_tree.greater_node, target, disc_index, skip, disc_func)
        if greater_node != None and worst_disc > greater_node.discrimination(disc_index, disc_func):
            worst_node = greater_node
            worst_disc = greater_node.discrimination(disc_index, disc_func)
    # if worst_disc == 999:
    #     print('no children')
    return worst_node

def get_bad_nodes_root_method(dec_tree, root_data=None, disc_func=disc_func.disc_fast, threshold=0, timeout=5, top=3, prop_thresh=0.4):
    if root_data == None: # dec tree is root
        root_data = dec_tree.disc_data
    leafs = dec_tree.get_leafs()
    # print(root_data)
    # sum = np.zeros((2,2,2))
    # for leaf in leafs:
    #     sum += leaf.disc_data
    # print(sum)
    best_leafs, num_data = _get_bad_nodes_root_method(np.copy(root_data), leafs, 0, disc_func, threshold, timeout, top, completed=[], id_sum=0, root_size=np.sum(root_data), prop_thresh=prop_thresh)
    print(num_data, np.sum(dec_tree.disc_data))
    # for leaf in best_leafs:
    #     print(np.sum(leaf.disc_data))
    return dec_tree.simplify_children(best_leafs)

    # combine leafs

def get_bad_nodes_root_method_bfs(dec_tree, root_data=None, disc_func=disc_func.disc_fast, threshold=0, timeout=5, top=3, prop_thresh=0.4):
    start_time = time.time()
    queue = []
    root_data = dec_tree.disc_data
    root_size = np.sum(root_data)
    root_leafs = dec_tree.get_leafs()
    best_leafs = []
    best_score = float('inf')
    best_disc = 0 # not best disc
    completed_id = [np.sum(root_leafs)]
    queue.append((root_data, root_leafs, 0))
    count = 0
    while len(queue) > 0:
        if time.time()-start_time >= timeout:
            print('timeout')
            break
        count += 1
        data, leafs, score = queue.pop(0)
        new_results = _get_bad_nodes_root_method_bfs(data, leafs, score, disc_func, top)
        for new_result in new_results:
            new_data, new_leafs, new_score = new_result
            disc = disc_func(new_data)
            if disc >= threshold:
                # print('root_disc:',disc, new_score/root_size)
                if new_score < best_score: # minimize
                    best_leafs = []
                    for leaf in root_leafs:
                        if leaf not in new_leafs:
                            best_leafs.append(leaf)
                    best_score = new_score
                    best_disc = disc
            else:
                if len(new_leafs) > 1:
                    id_sum = np.sum(new_leafs)
                else:
                    id_sum = -1
                if not(1-(np.sum(new_data)/root_size) > prop_thresh or len(leafs)==0 or disc<=-100 or id_sum in completed_id):
                    queue.append(new_result)
                    completed_id.append(id_sum)
    print('bad_nodes took:', time.time()-start_time, 'seconds', count, 'iterations')
    print('root_disc:',best_disc, best_score)
    return dec_tree.simplify_children(best_leafs)

def _get_bad_nodes_root_method_bfs(root_data, leafs, increment_score, disc_func, top):
    # requires: [(new data, available_leafs, sort_score)]
    root_disc = disc_func(root_data)
    # root_acc = (np.sum(root_data[:,0,0])+np.sum(root_data[:,1,1]))/np.sum(root_data[:,:,:])
    remaining_data = []
    delta_disc = []
    new_data = []
    for leaf in leafs:
        new_root_data = root_data-leaf.disc_data
        new_data.append(new_root_data)
        delta_disc.append(disc_func(new_root_data)-root_disc)
    sorted_leafs = list(zip(leafs, new_data))
    sorted_leafs = [elem for i,elem in sorted(enumerate(sorted_leafs), 
                                          key=lambda x:delta_disc[x[0]], reverse=True)]
    for i in range(top):
        # could add must be positive disc
        if i >= len(sorted_leafs):
            break
        new_root_data = sorted_leafs[i][1]
        new_leafs = []
        for j in range(len(sorted_leafs)):
            if i != j:
                new_leafs.append(sorted_leafs[j][0])
        # new_leafs.pop(0)
        # new_increment_score = increment_score+np.sum(sorted_leafs[i][0].disc_data) # total datapoints
        new_increment_score = -(np.sum(new_root_data[:,0,0])+np.sum(new_root_data[:,1,1]))/np.sum(new_root_data[:,:,:]) # accuracy
        remaining_data.append((new_root_data, new_leafs, new_increment_score))
    return remaining_data

def _get_bad_nodes_root_method(root_data, leafs, num_data, disc_func, threshold, timeout, top, completed, id_sum, root_size, prop_thresh):
    start = time.time()
    root_disc = disc_func(root_data)
    root_acc = (np.sum(root_data[:,0,0])+np.sum(root_data[:,1,1]))/np.sum(root_data[:,:,:])
    if root_disc >= threshold:
        print('root_disc:',root_disc, 1-(np.sum(root_data)/root_size), root_acc)
        return [], -root_acc
    if not leafs or root_disc<=-100:
        return [], -1
    if 1-(np.sum(root_data)/root_size) > prop_thresh:
        return [], -1
    best_leafs = []
    least_data = float('inf')
    delta_disc = []
    new_data = []
    for leaf in leafs:
        new_root_data = root_data-leaf.disc_data
        new_data.append(new_root_data)
        delta_disc.append(disc_func(new_root_data)-root_disc)
    sorted_leafs = list(zip(leafs, new_data))
    sorted_leafs = [elem for i,elem in sorted(enumerate(sorted_leafs), 
                                          key=lambda x:delta_disc[x[0]], reverse=True)]
    for i in range(top):
        if time.time()-start > timeout:
            print('timeout')
            break
        # could add must be positive disc
        if i >= len(sorted_leafs):
            break
        new_id_sum = id_sum+sorted_leafs[i][0].unique_id
        if new_id_sum in completed:
            print('inside')
            continue
        new_root_data = sorted_leafs[i][1]
        
        new_leafs = []
        for sorted_leaf in sorted_leafs[i:]:
            new_leafs.append(sorted_leaf[0])
        new_leafs.pop(0)
        completed.append(new_id_sum)
        tmp_leafs, tmp_data = _get_bad_nodes_root_method(np.copy(new_root_data), new_leafs, num_data+np.sum(sorted_leafs[i][0].disc_data), disc_func, threshold, timeout-(time.time()-start), top, completed, new_id_sum,  root_size, prop_thresh)
        if tmp_data == -1:
            continue
        if tmp_data < least_data:
            tmp_leafs.append(sorted_leafs[i][0])
            least_data = tmp_data
            best_leafs = tmp_leafs
    if least_data == float('inf'):
        return [], -1
    return best_leafs, least_data


def get_bad_nodes_child_method(dec_tree, comp, disc_index, skip, disc_func=disc_func.discrimination):
    '''
    comp: discrimination score to be identified as bad (None sets comp to root's discrimination)
    disc_index: index of discriminatory attribute
    disc_func: function used for discrimination
    stats: boolean prints statistics about bad node
    get_bad_nodes_child_method: identifies bad node as one who's children are both labeled as bad which is when:
        discrimination is worse than comp (or is leaf and parent's discrimination is worse than comp)
    return: list of bad nodes
    '''
    bad_nodes = []
    if comp == None:
        comp = dec_tree.discrimination(disc_index, disc_func)
    if (dec_tree.less_node.is_leaf() and dec_tree.discrimination(disc_index, disc_func) < comp) or (not dec_tree.less_node.is_leaf() and dec_tree.less_node.discrimination(disc_index) < comp):
        if (dec_tree.greater_node.is_leaf() and dec_tree.discrimination(disc_index, disc_func) < comp) or (not dec_tree.greater_node.is_leaf() and dec_tree.greater_node.discrimination(disc_index) < comp):
            return [dec_tree]
    if not dec_tree.less_node.is_leaf():
        bad_nodes.extend(get_bad_nodes_child_method(dec_tree.less_node, comp, disc_index, skip, disc_func))
    if not dec_tree.greater_node.is_leaf():
        bad_nodes.extend(get_bad_nodes_child_method(dec_tree.greater_node, comp, disc_index, skip, disc_func))
    return bad_nodes