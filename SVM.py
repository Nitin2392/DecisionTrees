import pandas as pd
import os
data_2c = pd.read_csv(os.path.expanduser("~")+"\\Biomechanical_Data_column_2C_weka.csv")
# print(data_2c)


from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import export_graphviz
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score
x = data_2c[data_2c.columns[:6]]
# So, x will have the feature vectors and y will have the class column
y = data_2c['class']
print(x)
print(y)

train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.68)
# Decision Tree Model for 10 leaf nodes
svm_predict = svm.SVC(kernel="linear", C=2.15).fit(train_x, train_y)
y_pred = svm_predict.predict(test_x)
print("Accuracy: %.3f" %accuracy_score(test_y, y_pred))
print("Precision: %.3f" %precision_score(test_y, y_pred, average="macro"))
print("Recall: %.3f" %recall_score(test_y, y_pred, average="macro"))

# Calculate Support Vectors
support_vectors = svm_predict.support_vectors_
print("Support Vectors : \n ", support_vectors)

# Calculate Support Vectors Indices
support_vectors_indices = svm_predict.support_
print("Support Vector Indices \n", support_vectors_indices)

# Calculate Count of Support Vectors for Each Class
support_vectors_count = svm_predict.n_support_
print("Count of Support Vectors \n", support_vectors_count)
print("Abnormal Class ", support_vectors_count[0])
print("Normal Class", support_vectors_count[1])

# To compute statistics for this SVM, we shall create a data frame
# and push these SVM values into this data frame.
# We shall thus get 6 columns in the data frame on which we shall conduct the analysis ( mean, median, mode)


# Confusion Matrix

import matplotlib.pyplot as plt
import pylab as pl

print("----Come down here----")

for i in range(5):
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.68)
    dft = pd.DataFrame(train_x)
    print("SD:", dft.describe().std())
    print("Mean:", dft.describe().mean())
    svm_cases = svm.SVC(kernel="linear", C=2.15).fit(train_x, train_y)
    y_pred_10 = svm_cases.predict(test_x)
    print("Accuracy: %.3f" %accuracy_score(test_y, y_pred_10))
    print("Precision: %.3f" %precision_score(test_y, y_pred_10, average="macro"))
    print("Recall: %.3f" %recall_score(test_y, y_pred_10, average="macro"))
    print(metrics.confusion_matrix(test_y, y_pred_10))


labels = ['Normal', 'Abnormal']
cm = metrics.confusion_matrix(test_y, y_pred_10)
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

# Part 2 - Use Radial Basis Function
print("\n \n \n Radial Basis Function Method Starts here \n \n \n ")
import pandas as pd
import os
data_2c = pd.read_csv(os.path.expanduser("~")+"\\Biomechanical_Data_column_2C_weka.csv")
# print(data_2c)


from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import export_graphviz
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score
x = data_2c[data_2c.columns[:6]]
# So, x will have the feature vectors and y will have the class column
y = data_2c['class']
print(x)
print(y)

train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.68)
# Decision Tree Model for 10 leaf nodes
svm_predict = svm.SVC(kernel="rbf", C=2.15, gamma=0.001).fit(train_x, train_y)
y_pred = svm_predict.predict(test_x)
print("Accuracy: %.3f" %accuracy_score(test_y, y_pred))
print("Precision: %.3f" %precision_score(test_y, y_pred, average="macro"))
print("Recall: %.3f" %recall_score(test_y, y_pred, average="macro"))

# Calculate Support Vectors
support_vectors = svm_predict.support_vectors_
print("Support Vectors for RBF: \n ", support_vectors)

# Calculate Support Vectors Indices
support_vectors_indices = svm_predict.support_
print("Support Vector Indices RBF\n", support_vectors_indices)

# Calculate Count of Support Vectors for Each Class
support_vectors_count = svm_predict.n_support_
print("Count of Support Vectors RBF\n", support_vectors_count)
print("Abnormal Class RBF", support_vectors_count[0])
print("Normal Class RBF", support_vectors_count[1])

# To compute statistics for this SVM, we shall create a data frame
# and push these SVM values into this data frame.
# We shall thus get 6 columns in the data frame on which we shall conduct the analysis ( mean, median, mode)


# Confusion Matrix

import matplotlib.pyplot as plt
import pylab as pl


for i in range(5):
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.68)
    dft = pd.DataFrame(train_x)
    print("SD:", dft.describe().std())
    print("Mean:", dft.describe().mean())
    svm_cases = svm.SVC(kernel="rbf", C=2.15, gamma=0.001).fit(train_x, train_y)
    y_pred_10 = svm_cases.predict(test_x)
    print("Accuracy: %.3f "  %accuracy_score(test_y, y_pred_10))
    print("Precision: %.3f" %precision_score(test_y, y_pred_10, average="macro"))
    print("Recall: %.3f" %recall_score(test_y, y_pred_10, average="macro"))
    print(metrics.confusion_matrix(test_y, y_pred_10))


labels = ['Normal', 'Abnormal']
cm = metrics.confusion_matrix(test_y, y_pred_10)
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