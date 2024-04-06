from utils import get_selected_features, load_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

individual = [1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]

data = load_data()

selected_train_f = get_selected_features(individual, data[0])
selected_test_f = get_selected_features(individual, data[2])

clf = DecisionTreeClassifier()
clf = clf.fit(selected_train_f, data[1])

pred = clf.predict(selected_test_f)
print(classification_report(data[3], pred))