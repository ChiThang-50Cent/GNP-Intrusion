import pandas as pd

def get_selected_features(selected_features, all_features):
    list_id = [i for i, e in enumerate(selected_features) if e == 1]
    return all_features[:, list_id]

def get_predicted_labels(class_prob):
    predicted_labels = []
    
    for p in class_prob:
        predicted_labels.append(p.argmax())
            
    return predicted_labels
    
def get_class_miss_percentages(true_labels, predicted_labels):
    c0_missed = 0
    c0_tot = 0
    c1_missed = 0
    c1_tot = 0

    
    for i in range(len(true_labels)):
        if true_labels[i] == 0:
            c0_tot += 1
            if true_labels[i] != predicted_labels[i]:
                c0_missed += 1
        
        if true_labels[i] == 1:
            c1_tot += 1
            if true_labels[i] != predicted_labels[i]:
                c1_missed += 1
        
    c0_miss_percent = (100.00 * c0_missed) / c0_tot  
    c1_miss_percent = (100.00 * c1_missed) / c1_tot
    
    if c0_miss_percent <= 1:
        c0_miss_percent = 1
    if c1_miss_percent <= 1:
        c1_miss_percent = 1
        
    print ("Missed samples for each class: ", c0_missed, c1_missed)
    
    return c0_miss_percent, c1_miss_percent
    
def load_data():

    train = pd.read_csv('./train.csv')
    training_data = train.values
    training_features = training_data[:, :-1]
    training_labels = training_data[:, -1]

    test = pd.read_csv('./test.csv')
    test_data = test.values       
    test_features = test_data[:, :-1]
    test_labels = test_data[:, -1]
        
    cost = pd.read_csv('./cost.csv')
    costs = cost.values[0]
        
    return training_features, training_labels, test_features, test_labels, costs


if __name__ == "__main__":
    data = load_data()
    d = data[0]

    list_ind = [i for i, e in enumerate([0,1,1]) if e == 1]
    print(d[:, list_ind].shape)
