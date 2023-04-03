import numpy as np

def discrimination(data, discriminatory_index):
    return statParity_equalOdds(data, discriminatory_index)
    # return statisticalParity(data, discriminatory_index)

def accuracy(data):
    if len(data) == 0:
        return 0
    correct = 0
    for datum in data:
        if datum[-1] == datum[-2]:
            correct+=1
    return correct / len(data)

def statisticalParity(data, protectedIndex):
    '''
    data: 2d array-like
    protectedIndex: int index of the protected class
    return difference of positive predictions of protected class
    '''
    if type(data) == list:
        data = np.array(data)
    if data.size == 0:
        return 0
    protectedClass = data[np.where(data[:,protectedIndex]==0)]
    elseClass = data[np.where(data[:,protectedIndex]!=0)]
 
    if len(protectedClass) == 0 or len(elseClass) == 0:
        return 0
    else:
        protectedProb = np.count_nonzero(protectedClass[:,-1]) / len(protectedClass)
        elseProb = np.count_nonzero(elseClass[:,-1]) / len(elseClass)
 
    return protectedProb-elseProb

def equalizedOdds(data, protectedIndex):
    '''
    data: 2d array-like
    protectedIndex: int index of the protected class
    return difference of positive predictions of protected class (given true pos)
    If Pos, they are equally as likely to be Pos predicted
    '''
    if type(data) == list:
        data = np.array(data)
    if data.size == 0:
        return 0
    protectedClass = data[np.where(data[:,protectedIndex]==0)]
    protectedClass = protectedClass[np.where(protectedClass[:,-1]==protectedClass[:,-2])]
    elseClass = data[np.where(data[:,protectedIndex]!=0)]
    elseClass = elseClass[np.where(elseClass[:,-1]==elseClass[:,-2])]
 
    if len(protectedClass) == 0 or len(elseClass) == 0:
        return 0
    else:
        protectedProb = np.count_nonzero(protectedClass[:,-1]) / len(protectedClass)
        elseProb = np.count_nonzero(elseClass[:,-1]) / len(elseClass)

    return protectedProb-elseProb

def statParity_equalOdds(data, protectedIndex):
    '''
    data: 2d array-like
    protectedIndex: int index of the protected class
    return average of statistical parity and Equalized odds
    '''
    if type(data) == list:
        data = np.array(data)
    if data.size == 0:
        return 0
    protectedClass = data[np.where(data[:,protectedIndex]==0)]
    elseClass = data[np.where(data[:,protectedIndex]!=0)]
 
    if len(protectedClass) == 0 or len(elseClass) == 0:
        return 0
    else:
        protectedProb = np.count_nonzero(protectedClass[:,-1]) / len(protectedClass)
        elseProb = np.count_nonzero(elseClass[:,-1]) / len(elseClass)
 
    statParity = protectedProb-elseProb

    protectedClass = protectedClass[np.where(protectedClass[:,-1]==protectedClass[:,-2])]
    elseClass = elseClass[np.where(elseClass[:,-1]==elseClass[:,-2])]
 
    if len(protectedClass) == 0 or len(elseClass) == 0:
        return 0
    else:
        protectedProb = np.count_nonzero(protectedClass[:,-1]) / len(protectedClass)
        elseProb = np.count_nonzero(elseClass[:,-1]) / len(elseClass)

    equalOdds = protectedProb-elseProb

    return (equalOdds + statParity)/2