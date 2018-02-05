import pandas as pd
import os
data_2c = pd.read_csv(os.path.expanduser("~")+"\\Biomechanical_Data_column_2C_weka.csv")

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, accuracy_score

x = data_2c[data_2c.columns[:6]]
# So, x will have the feature vectors and y will have the class column
y = data_2c['class']
xnew = x

# Boundary values - Binning the column into 4 bins/categories. Value.counts() count of each bin
# Here the output will print the range of values inside each bin
print(pd.qcut(xnew['pelvic_incidence'], 4).value_counts())
print(pd.qcut(xnew['pelvic_tilt numeric'], 4).value_counts())
print(pd.qcut(xnew['pelvic_radius'], 4).value_counts())
print(pd.qcut(xnew['lumbar_lordosis_angle'], 4).value_counts())
print(pd.qcut(xnew['sacral_slope'], 4).value_counts())
print(pd.qcut(xnew['degree_spondylolisthesis'], 4).value_counts())

# Convert values
# Here, we are assigning these bins as new values to the column.
# If, in the original file, a point was 78.90 and this point was in the 2 category. Then, this code will
# overwrite that cell's value with 2.
xnew['lumbar_lordosis_angle'] = pd.qcut(xnew['lumbar_lordosis_angle'], 4, labels=False)
xnew['pelvic_incidence'] = pd.qcut(xnew['pelvic_incidence'], 4, labels=False)
xnew['pelvic_tilt numeric'] = pd.qcut(xnew['pelvic_tilt numeric'], 4, labels=False)
xnew['pelvic_radius'] = pd.qcut(xnew['pelvic_radius'], 4, labels=False)
xnew['sacral_slope'] = pd.qcut(xnew['sacral_slope'], 4, labels=False)
xnew['degree_spondylolisthesis'] = pd.qcut(xnew['degree_spondylolisthesis'], 4, labels=False)

# Display count of new values
# Here, we are printing the new value
print(xnew['pelvic_incidence'].value_counts())
print(xnew['pelvic_radius'].value_counts())
print(xnew['pelvic_tilt numeric'].value_counts())
print(xnew['lumbar_lordosis_angle'].value_counts())
print(xnew['sacral_slope'].value_counts())
print(xnew['degree_spondylolisthesis'].value_counts())



