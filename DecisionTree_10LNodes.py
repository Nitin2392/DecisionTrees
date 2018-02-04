import pandas as pd
import os
data_2c = pd.read_csv(os.path.expanduser("~")+"\\Biomechanical_Data_column_2C_weka.csv")
# print(data_2c)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
x = data_2c[data_2c.columns[:6]]
# So, x will have the feature vectors and y will have the class column
y = data_2c['class']
print(x)
print(y)
# Decision Tree Model for 10 leaf nodes
train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.68)

dt_10 = DecisionTreeClassifier(min_samples_leaf=10, random_state=10).fit(train_x, train_y)
y_pred_10 = dt_10.predict(test_x)
print("\n Data below is for 10 leaf nodes \n")
print("Accuracy: %.3f" %accuracy_score(test_y, y_pred_10))
print("Precision: %.3f" %precision_score(test_y, y_pred_10, average="weighted"))
print("Recall: %.3f" %recall_score(test_y, y_pred_10, average="weighted"))

# Decision Tree Model for 15 leaf nodes

dt_15 = DecisionTreeClassifier(min_samples_leaf=10, random_state=10).fit(train_x, train_y)
y_pred_15 = dt_15.predict(test_x)
print("\n Data below is for 15 leaf nodes \n")
print("Accuracy: %.3f" %accuracy_score(test_y, y_pred_15))
print("Precision: %.3f" %precision_score(test_y, y_pred_15, average="weighted"))
print("Recall: %.3f" %recall_score(test_y, y_pred_15, average="weighted"))

# Decision Tree Model for 20 leaf nodes

dt_20 = DecisionTreeClassifier(min_samples_leaf=10, random_state=10).fit(train_x, train_y)
y_pred_20 = dt_20.predict(test_x)
print("\n Data below is for 20 leaf nodes \n")
print("Accuracy: %.3f" %accuracy_score(test_y, y_pred_20))
print("Precision: %.3f" %precision_score(test_y, y_pred_20, average="weighted"))
print("Recall: %.3f" %recall_score(test_y, y_pred_20, average="weighted"))

# Decision Tree Model for 25 leaf nodes

dt_25 = DecisionTreeClassifier(min_samples_leaf=10, random_state=10).fit(train_x, train_y)
y_pred_25 = dt_25.predict(test_x)
print("\n Data below is for 25 leaf nodes \n")
print("Accuracy: %.3f" %accuracy_score(test_y, y_pred_25))
print("Precision: %.3f" %precision_score(test_y, y_pred_25, average="weighted"))
print("Recall: %.3f" %recall_score(test_y, y_pred_25, average="weighted"))