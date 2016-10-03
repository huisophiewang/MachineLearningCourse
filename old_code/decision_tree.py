import numpy as np
import  math

def entropy_before(data):
    n = data.shape[0]
    d = data.shape[1]
    n0 = 0
    n1 = 0
    for sample in data:
        if sample[d-1] == 0:
            n0 += 1
        elif sample[d-1] == 1:
            n1 += 1
            
    p0 = n0 / n
    p1 = n1 / n
    
    if p0 == 0 | p1 == 0:
        return 0
    else:   
        return p0 * math.log(1/p0, 2) + p1 * math.log(1/p1, 2)
            
def entropy_after_split(data, split_feature, split_value):
    n = data.shape[0]
    d = data.shape[1]
    
    data = data[np.argsort(data[:, split_feature])]
    nleft0 = 0
    nleft1 = 0
    nright0 = 0
    nright1 = 0

    for sample in data:
        if sample[split_feature] <= split_value:
            if sample[d-1] == 0:
                nleft0 += 1
            elif sample[d-1] == 1:
                nleft1 += 1
        else:
            if sample[d-1] == 0:
                nright0 += 1
            elif sample[d-1] == 1:
                nright1 += 1          
    
    nleft = nleft0 + nleft1
    nright = nright0 + nright1
    
    pleft = nleft / n
    pright = nright / n
    
    pleft0 = nleft0 / nleft
    pleft1 = nleft1 / nleft
    
    pright0 = nright0 / nright
    pright1 = nright1 / nright
    
    if pleft0 == 0 or pleft1 == 0:
        entropy_left = 0
    else:
        entropy_left = pleft * (pleft0 * math.log(1/pleft0, 2) + pleft1 * math.log(1/pleft1, 2))
        
    if pright0 == 0 or pright1 == 0:
        entropy_right = 0
    else:
        entropy_right = pright * (pright0 * math.log(1/pright0, 2) + pright1 * math.log(1/pright1, 2))
    
    entropy = entropy_left + entropy_right
    
    return  entropy
    
def majority_vote(data):
    
    n = data.shape[0]
    d = data.shape[1]
    
    num_1 = 0
    num_0 = 0
    for sample in data:
        if sample[d-1] == 1:
            num_1 += 1
        else:
            num_0 += 1
            
    if num_1 > num_0:
        return 1
    else:
        return 0

def run_through_tree(data, tree):

    n = data.shape[0]
    d = data.shape[1]
    
    num_correct = 0
    
    
    for sample in data:
        # reset tree to the original regression tree
        run_tree = tree
        while 'label' not in run_tree:
            col = run_tree['feature']
            if sample[col] <= run_tree['value']:
                run_tree = run_tree['left']
            else:
                run_tree = run_tree['right']
         
        predicted_value = run_tree['label']
        if predicted_value == sample[d-1]:
            num_correct += 1
        print '-----'
        print sample[d-1]
        print run_tree['label']



    accuracy = num_correct / n 
    return accuracy

def build_decision_tree(data, depth):
    tree = {}
    
    n = data.shape[0]
    d = data.shape[1]
    

    if depth == 0:
        tree['label'] = majority_vote(data)
    elif n < 10:
        tree['label'] = data[:, d-1].mean()

    else:
        
        feature_min_entropy = []
        # for each feature, not y label
        for j in range(d-1):
          
            data = data[np.argsort(data[:, j])]

            min_entropy = []
            # for each unique value, find the mid split value that has min error
                
            for i in range(1, n):
                
                if data[i, j] > data[i-1, j]:
                    # use mid point to split
                    #split_value = (data[i, j] + data[i-1, j]) / 2.0
                    split_value = data[i-1, j]
    
                    #split_error = data[:i, d-1].var() + data[i:, d-1].var()
                    # total square error of left and right, (variance is divided by n, scale is 2 times after split, don't use here)
                    
                    #split_error = np.sum(np.square(data[:i, d-1] - data[:i, d-1].mean())) + np.sum(np.square(data[i:, d-1] - data[i:, d-1].mean()))
                    entropy_after = entropy_after_split(data, j, split_value)
        
                    min_entropy.append((j, split_value, entropy_after))
             
                  
            if min_entropy:    
                feature_min_entropy.append(sorted(min_entropy, key=lambda x: x[2])[0])
            
    
        best = sorted(feature_min_entropy, key=lambda x: x[2])[0]
        best_feature = best[0]
        best_value = best[1]
        best_entropy = best[2]
        
        print best
        
        entropy_drop = entropy_before(data) - best_entropy
        
        

        if entropy_drop > 0:
    
            left_index = np.array([row[best_feature] <= best_value for row in data])
            left_data = data[left_index, :]

            right_index = np.array([row[best_feature] > best_value for row in data])
            right_data = data[right_index, :]

            tree['feature'] = best_feature
            tree['value'] = best_value
            tree['left'] = build_decision_tree(left_data, depth-1)
            tree['right'] = build_decision_tree(right_data, depth-1)
        else:
            tree['label'] = majority_vote(data)
            
    return tree


if __name__ == '__main__':
    data = np.loadtxt("spambase.data.txt", delimiter=',')
    print data.shape
    
    n = data.shape[0]
    d = data.shape[1]
    k = 10
    
    subarray = np.array_split(data, 10)
    train_data = np.array([]).reshape(0, d)
    train_data = np.vstack((train_data, subarray[1]))
    #for i in range(k):
    i= 0
    test_data = subarray[i]
 
    train_data = np.array([]).reshape(0, d)
    for j in range(k):
        if j != i:
            train_data = np.vstack((train_data, subarray[j]))
            
    print train_data.shape
    dec_tree = build_decision_tree(train_data, 1)
    print dec_tree
    acc = run_through_tree(test_data, dec_tree)
    print acc
    
    #acu = run_through_tree(data[:n%k, :], dec_tree)
    
    
    
    
        