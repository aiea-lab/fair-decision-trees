import numpy as np
import decision_tree

def predictive_similarity(before, after):
    '''
    before: list of numpy arrays containing attributes, true class, and predictions
    after: list of numpy arrays containing attributes, true class, and predictions
    predictive_similarity: compares the predictions and checks how many stayed the same
    return: ratio of unchanged predictions
    '''
    initial_preds = []
    for data in before:
        initial_preds.append(data[-1])
    initial_preds = np.array(initial_preds)
    final_preds = []
    for data in after:
        final_preds.append(data[-1])
    final_preds = np.array(final_preds)
    return np.average(initial_preds==final_preds)    

def naive_discounted_similarity(before, after, discount=0.7, min_feature_score=0.5):
    if before == None or after == None:
        return 0
    if before.is_leaf() and after.is_leaf():
        return before.leaf == after.leaf
    if before.is_leaf() or after.is_leaf():
        return 0
    naive_score = 0
    if before.feature == after.feature:
        naive_score = min_feature_score
        if max([before.threshold, after.threshold])==0:
            naive_score += 0
        else:
            naive_score += 0.5 * min(before.threshold, after.threshold)/max([before.threshold, after.threshold])
    return (1-discount)*naive_score+discount*0.5*(naive_discounted_similarity(before.less_node, after.less_node, discount, min_feature_score)+naive_discounted_similarity(before.greater_node, after.greater_node, discount, min_feature_score))