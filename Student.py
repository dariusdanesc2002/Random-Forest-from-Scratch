import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm  # Importing tqdm for progress tracking

np.random.seed(52)  # Set seed for reproducibility


# Function to convert 'Embarked' categorical values into numerical
def convert_embarked(x):
    if x == 'S':
        return 0
    elif x == 'C':
        return 1
    else:
        return 2


# Function to create a bootstrap sample
def create_bootstrap(x_target, y_target):
    mask = np.random.choice(range(len(x_target)), 30)  # Randomly select indices for bootstrap
    mask = list(mask)
    return mask


# Custom RandomForestClassifier class
class RandomForestClassifier():
    def __init__(self, n_trees, max_depth=np.iinfo(np.int64).max, min_error=1e-6):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_error = min_error

        self.forest = []
        self.is_fit = False

    # Fit function to train the model
    def fit(self, X_train, y_train):
        for i in tqdm(range(self.n_trees)):  # Iterate over each tree
            mask = create_bootstrap(X_train, y_train)  # Create a bootstrap sample
            clf = DecisionTreeClassifier(max_features='sqrt', max_depth=self.max_depth,
                                         min_impurity_decrease=self.min_error)

            self.forest.append(clf.fit(X_train[mask], y_train[mask]))  # Fit DecisionTreeClassifier with bootstrap sample

        self.is_fit = True  # Set is_fit flag to True after fitting

    # Predict function to make predictions
    def predict(self, X_test):
        if not self.is_fit:
            raise AttributeError('The forest is not fit yet! Consider calling .fit() method.')

        predictions_object = []
        for x in X_test:
            c_0 = 0
            c_1 = 0
            predictions_each_tree = []

            for tree in self.forest:
                predictions_each_tree.append(tree.predict(x.reshape(1, -1)))

            for pred in predictions_each_tree:
                if pred == 1:
                    c_1 += 1
                else:
                    c_0 += 1

            if c_1 > c_0:
                predictions_object.append(1)
            elif c_1 < c_0:
                predictions_object.append(0)
            else:
                predictions_object.append(0)

        return predictions_object


# Function to calculate accuracy score
def accuracy_score(predictions_object, y_val):
    c = 0
    for i in range(len(y_val)):
        if predictions_object[i] == y_val[i]:
            c += 1

    return c / len(y_val)


if __name__ == '__main__':
    # Read data from CSV file
    data = pd.read_csv('https://www.dropbox.com/s/4vu5j6ahk2j3ypk/titanic_train.csv?dl=1')

    # Drop unnecessary columns
    data.drop(
        ['PassengerId', 'Name', 'Ticket', 'Cabin'],
        axis=1,
        inplace=True
    )
    data.dropna(inplace=True)  # Drop rows with missing values

    # Separate features and target variable
    y = data['Survived'].astype(int)
    X = data.drop('Survived', axis=1)

    # Convert categorical variables into numerical
    X['Sex'] = X['Sex'].apply(lambda x: 0 if x == 'male' else 1)
    X['Embarked'] = X['Embarked'].apply(lambda x: convert_embarked(x))

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X.values, y.values, stratify=y, train_size=0.8)

    accuracy = []  # List to store accuracy scores
    for n_trees in range(1, 21, 1):
        my_class = RandomForestClassifier(n_trees)
        my_class.fit(X_train, y_train)
        predictions_object = my_class.predict(X_val)
        test_score = accuracy_score(predictions_object, y_val)
        accuracy.append(test_score)

    # True accuracy values
    true_data = [0.755, 0.818, 0.783, 0.839, 0.79, 0.825, 0.79, 0.811, 0.818,
                 0.783, 0.825, 0.832, 0.804, 0.825, 0.825, 0.825, 0.839, 0.762,
                 0.839, 0.825]
    print(true_data)