import numpy as np
import math
from pprint import pprint


def run_through_tree(data, tree):

    n = data.shape[0]
    d = data.shape[1]
    
    error_sum = 0
    run_tree = tree
    
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
        print '-----'
        print sample[d-1]
        print run_tree['label']
        error_sum += math.pow((predicted_value - sample[d-1]), 2) 


    mse = error_sum / n 
    return mse



def build_regression_tree(data, depth):
    tree = {}
    
    n = data.shape[0]
    d = data.shape[1]
    

    if depth == 0:
        tree['label'] = data[:, d-1].mean()
    elif n < 10:
        tree['label'] = data[:, d-1].mean()

    else:
        
        feature_min_error = []
        # for each feature, not y label
        for j in range(d-1):
          
            data = data[np.argsort(data[:, j])]

            value_min_error = []
            # for each unique value, find the mid split value that has min error
                
            for i in range(1, n):
                
                if data[i, j] > data[i-1, j]:
                    # use mid point to split
                    #split_value = (data[i, j] + data[i-1, j]) / 2.0
                    split_value = data[i-1, j]
    
                    split_error = data[:i, d-1].var()*len(data[:i, d-1]) + data[i:, d-1].var()*len(data[i:, d-1])
                    # total square error of left and right, (variance is divided by n, scale is 2 times after split, don't use here)
                    
                    #split_error = np.sum(np.square(data[:i, d-1] - data[:i, d-1].mean())) + np.sum(np.square(data[i:, d-1] - data[i:, d-1].mean()))
        
                    value_min_error.append((j, split_value, split_error))
             
                  
            if value_min_error:    
                feature_min_error.append(sorted(value_min_error, key=lambda x: x[2])[0])
            
    
        best = sorted(feature_min_error, key=lambda x: x[2])[0]
        best_feature = best[0]
        best_value = best[1]
        best_error = best[2]
        
        print best
        
        error_before = np.sum(np.square(data[:, d-1] - data[:, d-1].mean()))

        if error_before - best_error > 100:
    
            left_index = np.array([row[best_feature] <= best_value for row in data])
            left_data = data[left_index, :]

            right_index = np.array([row[best_feature] > best_value for row in data])
            right_data = data[right_index, :]

            tree['feature'] = best_feature
            tree['value'] = best_value
            tree['left'] = build_regression_tree(left_data, depth-1)
            tree['right'] = build_regression_tree(right_data, depth-1)
        else:
            tree['label'] = data[:, d-1].mean()
            
    return tree



if __name__ == '__main__':
    training_data = np.loadtxt("housing_train.txt")
    testing_data = np.loadtxt("housing_test.txt")
    

    
    reg_tree = build_regression_tree(training_data, 20)
    
    pprint(reg_tree)
    
    train_error = run_through_tree(training_data, reg_tree)
    
    
    test_error = run_through_tree(testing_data, reg_tree)
    
    print train_error
    print test_error

    
    
