import pandas as pd
import os
data_2c = pd.read_csv(os.path.expanduser("~")+"\\Biomechanical_Data_column_2C_weka.csv")
# print(data_2c)

from sklearn.model_selection import train_test_split


# We need to split the data into feature vectors and classes
# Post that, we need to specify the percentage split for training data and testing data
# The basic principle of ML is that we train a decision model feeding it with training data
# and then test the model's accuracy and precision with the testing data
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.tree import export_graphviz

x = data_2c[data_2c.columns[:6]]
# So, x will have the feature vectors and y will have the class column
y = data_2c['class']
print(x)
print(y)
# Using the below command, we can split the data into training data and testing data.
# We are choosing 210/310 rows for training data and thus using that split (210/310 = 0.67) to
# push 0.67 * x into train_x and 0.67* y as train_y and the rest to test_x and test_y
train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.68)

print("train_x size :: ", train_x.shape)
print("train_y size :: ", train_y.shape)
print("test_x size :: ", test_x.shape)
print("test_y size :: ", test_y.shape)

# For the decision tree to stop, we need to specify some stopping conditions.
# The stopping conditions can be like the one specified below when the decision tree
# stops/terminates when the algorithm identifies 5 data points with the leaf node condition
# and this condition is a randomly chosen feature vector data point based on Gini/Entropy
dt = DecisionTreeClassifier(min_samples_leaf=5, random_state=10).fit(train_x, train_y)
# In the above command, we are building the model dt by specifying that the model should stop
# when 5 nodes match the lead node condition. Random state indicates that the model should not randomly
# change value state every time the algorithm is run.We are fitting the training data to the Classifier
y_pred = dt.predict(test_x)
# In the above command, we are using that generated model to predict the result of test_x
# test_y is the true value - The class division
print("Accuracy: %.3f" %accuracy_score(test_y, y_pred))
# Accuracy identifies how accurate the model dt predicted the test_x
print("Precision: %.3f" %precision_score(test_y, y_pred, average="weighted"))
# Precision is defined as the ratio of True Positive to the (TP + FP)
# i.e the ratio of the intersection of actual and precision to the actual positive values
print("Recall: %.3f" %recall_score(test_y, y_pred, average="weighted"))
# Recall is defined as the ratio of True Positive to the (TP + FN)
# i.e the ratio of the intersection of actual and precision to the predicted positive values
