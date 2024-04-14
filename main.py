from utils import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from random import randint, random

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

import argparse
import numpy as np

# Các hằng số này được sử dụng trong quá trình khởi tạo và xử lý cá thể (individuals).
N_JUDGE = 2
N_PROCESS = 2
N_NODE = 10
N_CON = 5
J_SUM = 4

# Đây là lớp Gen đại diện cho một cá thể trong quần thể (population).
# Nó có các thuộc tính như kiểu (type), các đặc trưng được chọn (cn),
# hàm xử lý (process_func), các đặc trưng (features), và độ dài (length). 
# Lớp này cũng có các phương thức setup, process_0, process_1 để khởi tạo và xử lý cá thể.
class Gen():
    num_node = 0

    def __init__(self, length):
        self.type = None # j = 0, p = 1
        self.cn = np.random.choice(N_NODE, N_CON, replace=False)
        self.process_func = None
        self.features = None
        self.length = len(length)

        self.p = [self.process_0, self.process_1]

    def setup(self):
        if np.random.randint(2) and Gen.num_node < N_NODE:
            self.type = 0
        else:
            self.type = 1
        
        if self.type:
            self.process_func = self.p[np.random.randint(N_PROCESS)]

        self.features = np.random.randint(2, size=self.length)

    def process_0(self):
        index = np.random.choice(self.features, 5)
        self.features[index] = 1

    def process_1(self):
        index = np.random.choice(self.features, 5)
        self.features[index] = 0
    
    def judge(self):
        pass

# Đây là hàm chính của chương trình. Nó thực hiện các bước sau:
# Tải dữ liệu huấn luyện và kiểm tra từ load_data.
# Khởi tạo quần thể ban đầu.
# Lặp qua các thế hệ và tiến hóa quần thể bằng cách gọi evolve.
# In ra lịch sử độ phù hợp trung bình của quần thể.
def main(model):
    training_features, training_labels, test_features, test_labels, costs = load_data()
    data = [training_features, training_labels, test_features, test_labels, costs]
    clf = model

    pop = population(20, training_features[0])

    fitness_history = []
    for _ in range(100):
        pop = evolve(pop, clf, data)
        pop_fitness = avg_fitness(pop, clf, data)
        fitness_history.append(pop_fitness)

        print("#" * 10, pop_fitness, "#" * 10)
        if pop_fitness <= 250:
            break

    print("Avg fitness history: ", fitness_history)

    # fittest_results(clf, training_features, training_labels, test_features, test_labels, costs)

# Các hàm này thực hiện các công việc cụ thể trong thuật toán di truyền:

# individual: tạo một cá thể mới.
def individual(length):
    # create an individual, which is binary repr. of selected feature

    gen = Gen(length=length)
    gen.setup()
    ind = gen.features

    return ind

# population: tạo một quần thể mới với số lượng cá thể nhất định.
def population(count, length):
    # create 'count' number of individuals
    return [individual(length) for _ in range(count)]

# fitness: đánh giá độ phù hợp của một cá thể dựa trên tỷ lệ phân loại sai và chi phí lựa chọn đặc trưng.
def fitness(individual, clf, data):
    # In ra dấu phân cách
    print("\n######################################################################")
    print("Individual: ", list(individual))

    # Tính toán chi phí lựa chọn đặc trưng (feature selection cost)
    fs_cost = feature_selection_cost(individual, data[4])

    # Lấy các đặc trưng được chọn cho tập huấn luyện và tập kiểm tra
    selected_train_f = get_selected_features(individual, data[0])
    selected_test_f = get_selected_features(individual, data[2])

    # Huấn luyện mô hình phân lớp với tập huấn luyện đã chọn đặc trưng
    clf = clf.fit(selected_train_f, data[1])

    # Dự đoán nhãn cho tập kiểm tra và tính toán xác suất dự đoán
    pred = clf.predict(selected_test_f)
    class_prob = clf.predict_proba(selected_test_f)
    predicted_labels = get_predicted_labels(class_prob)

    # In ra báo cáo phân lớp, độ chính xác trung bình và số lượng mẫu được phân lớp chính xác
    print(classification_report(data[3], pred))
    print("Mean accuracy: ", clf.score(selected_test_f, data[3]))
    print("No of correctly classified samples: ", accuracy_score(data[3], predicted_labels, normalize=False))

    # Tính toán tỷ lệ sai lầm cho mỗi lớp
    c1_miss_percent, c2_miss_percent = get_class_miss_percentages(data[3], predicted_labels)

    # Tính toán giá trị fitness dựa trên tỷ lệ sai lầm và chi phí lựa chọn đặc trưng
    f_result = int(c1_miss_percent * c2_miss_percent * (fs_cost // 10))

    # In ra thông tin về độ chính xác cho mỗi lớp, chi phí lựa chọn đặc trưng và giá trị fitness
    print("\nClass accuracies: \n", "class 0: ", (100 - c1_miss_percent), "%\nclass 1: ", (100 - c2_miss_percent), "%\n")
    print("Feature selection cost: ", fs_cost)
    print("Fitness: ", f_result)
    print("######################################################################\n")

    # Trả về giá trị fitness
    return f_result
# avg_fitness: tính độ phù hợp trung bình của quần thể.
def avg_fitness(pop, clf, data):
    # average fitness of a population
    tot_fitness = 0
    for i in pop:
        tot_fitness += fitness(i, clf, data)
    return tot_fitness / len(pop)

# evolve: thực hiện quá trình tiến hóa của quần thể bằng cách giữ lại các cá thể tốt nhất, đột biến,
# lai tạo để tạo ra thế hệ mới.
def evolve(
    pop, clf, data, retain_percentage=0.50, random_select=0.05, mutate_prob=0.01
):
    # Tính toán giá trị fitness cho mỗi cá thể trong quần thể
    f_values = [(fitness(i, clf, data), i) for i in pop]
    
    # Sắp xếp các cá thể theo thứ tự giảm dần của giá trị fitness
    individuals = [i[1] for i in sorted(f_values, key=lambda x: x[0])]
    
    # Giữ lại một tỷ lệ nhất định của cá thể tốt nhất làm cha mẹ cho thế hệ mới
    retain_length = int(len(pop) * retain_percentage)
    parents = individuals[:retain_length]

    # Ngẫu nhiên chọn thêm một số cá thể khác để tăng đa dạng
    for i in individuals[retain_length:]:
        if random_select > random():
            parents.append(i)

    # mutate (Đột biến)
    for i in parents:
        if mutate_prob > random():
            index_to_mutate = randint(0, len(i) - 1)
            i[index_to_mutate] = randint(0, 1)

    # crossover(Lai tạo)
    no_of_parents = len(parents)
    remaining_no_of_ind = len(pop) - no_of_parents
    children = []

    while len(children) < remaining_no_of_ind:
        male_index = randint(0, no_of_parents - 1)
        female_index = randint(0, no_of_parents - 1)

        if male_index != female_index:
            male = parents[male_index]
            female = parents[female_index]
            half = len(male) // 2
            child = np.concatenate([male[:half], female[half:]])
            children.append(child)

    parents.extend(children)
    return parents


def feature_selection_cost(selected_features, costs):
    return sum([c for s, c in zip(selected_features, costs) if s == 1])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="dst",
        type=str,
        help="""
    dts -> Decision Tree
    lr -> LogisticRegression
    gnb -> GaussianNB
    knn -> KNeighborsClassifier
    ada -> AdaBoostClassifier
    """,
    )

    args = parser.parse_args()

    model_name = args.model
    model = None

    if model_name == "dst":
        model = DecisionTreeClassifier()
    elif model_name == "lr":
        model = LogisticRegression(solver="saga")
    elif model_name == "gnb":
        model = GaussianNB()
    elif model_name == "knn":
        model = KNeighborsClassifier()
    elif model_name == "ada":
        model = AdaBoostClassifier()

    main(model)
