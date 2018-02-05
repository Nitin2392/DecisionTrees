import pandas as pd
import os
data_2c = pd.read_csv(os.path.expanduser("~")+"\\Biomechanical_Data_column_2C_weka.csv")
# print(data_2c)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, accuracy_score
x = data_2c[data_2c.columns[:6]]
# So, x will have the feature vectors and y will have the class column
y = data_2c['class']
print(x)
print(y)
# Decision Tree Model for 10 leaf nodes
train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.68)

dt_5 = DecisionTreeClassifier(min_samples_leaf=5, random_state=10).fit(train_x, train_y)
y_pred_5 = dt_5.predict(test_x)
print("\n Data below is for 5 leaf nodes \n")
print("Accuracy: %.3f" %accuracy_score(test_y, y_pred_5))
print("Precision: %.3f" %precision_score(test_y, y_pred_5, average="macro"))
print("Recall: %.3f" %recall_score(test_y, y_pred_5, average="macro"))
print("Recall: %.3f" %recall_score(test_y, y_pred_5, average="macro"))
export_graphviz (dt_5, out_file='out_5.dot')


dt_10 = DecisionTreeClassifier(min_samples_leaf=10, random_state=10).fit(train_x, train_y)
y_pred_10 = dt_10.predict(test_x)
print("\n Data below is for 10 leaf nodes \n")
print("Accuracy: %.3f" %accuracy_score(test_y, y_pred_10))
print("Precision: %.3f" %precision_score(test_y, y_pred_10, average="macro"))
print("Recall: %.3f" %recall_score(test_y, y_pred_10, average="macro"))
print("Recall: %.3f" %recall_score(test_y, y_pred_10, average="macro"))
export_graphviz (dt_10, out_file='out_10.dot')

# Decision Tree Model for 15 leaf nodes

dt_15 = DecisionTreeClassifier(min_samples_leaf=15, random_state=10).fit(train_x, train_y)
y_pred_15 = dt_15.predict(test_x)
print("\n Data below is for 15 leaf nodes \n")
print("Accuracy: %.3f" %accuracy_score(test_y, y_pred_15))
print("Precision: %.3f" %precision_score(test_y, y_pred_15, average="macro"))
print("Recall: %.3f" %recall_score(test_y, y_pred_15, average="macro"))
print("Recall: %.3f" %recall_score(test_y, y_pred_15, average="macro"))
export_graphviz (dt_15, out_file='out_15.dot')

# Decision Tree Model for 20 leaf nodes

dt_20 = DecisionTreeClassifier(min_samples_leaf=20, random_state=10).fit(train_x, train_y)
y_pred_20 = dt_20.predict(test_x)
print("\n Data below is for 20 leaf nodes \n")
print("Accuracy: %.3f" %accuracy_score(test_y, y_pred_20))
print("Precision: %.3f" %precision_score(test_y, y_pred_20, average="macro"))
print("Recall: %.3f" %recall_score(test_y, y_pred_20, average="macro"))
export_graphviz (dt_20, out_file='out_20.dot')

# Decision Tree Model for 25 leaf nodes

dt_25 = DecisionTreeClassifier(min_samples_leaf=25, random_state=10).fit(train_x, train_y)
y_pred_25 = dt_25.predict(test_x)
print("\n Data below is for 25 leaf nodes \n")
print("Accuracy: %.3f" %accuracy_score(test_y, y_pred_25))
print("Precision: %.3f" %precision_score(test_y, y_pred_25, average="macro"))
print("Recall: %.3f" %recall_score(test_y, y_pred_25, average="macro"))
# The below code exports the tree structure to the out.dot file.
export_graphviz (dt_25, out_file='out25.dot')

# Comparing the true and predicted response values
print('True: ', test_y.values[0:25])
print('False:', (y_pred_25[0:25]))

# Confusion Matrix

import matplotlib.pyplot as plt
import pylab as pl


print("----Come down here----")

for i in range(5):
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.68)
    dft = pd.DataFrame(train_x)
    print("SD:", dft.describe().std())
    print("Mean:", dft.describe().mean())
    dt_10 = DecisionTreeClassifier(min_samples_leaf=10, random_state=10).fit(train_x, train_y)
    y_pred_10 = dt_10.predict(test_x)
    print("\n Data below is for 10 leaf nodes \n")
    print("Accuracy: %.3f" %accuracy_score(test_y, y_pred_10))
    print("Precision: %.3f" %precision_score(test_y, y_pred_10, average="macro"))
    print("Recall: %.3f" %recall_score(test_y, y_pred_10, average="macro"))
    print(metrics.confusion_matrix(test_y, y_pred_10))


labels = ['Normal', 'Abnormal']
cm = metrics.confusion_matrix(test_y, y_pred_25)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
pl.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
pl.xlabel('Predicted')
pl.ylabel('True')
pl.show()

