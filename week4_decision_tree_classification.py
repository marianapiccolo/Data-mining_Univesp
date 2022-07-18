# -*- coding: utf-8 -*-
"""week4_decision_tree_classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12UOxFT20OXO4-OR1hYk3-CkRcIG8wA6o

Data Mining - Week 4:

 Supervised Classification Exercise


In this exercise, we will do a supervised classification task using decision trees.


- Decision whether or not to go to comedy shows

- Evaluate classification accuracy - wine quality

Libraries:

Pandas;
Scikit learn;
Matplotlib

Data:

Comedy shows: https://www.kaggle.com/mruanova/decision-tree-for-comedy-shows-w3schools/data

Quality of wines:
https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
"""

from google.colab import files
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""# Building a classification model """

uploaded = files.upload()

comedians = pd.read_csv('shows.csv', sep = ',')

comedians

# Column Go: Go to the concert or not

# Column Nationality: N - Person not native to UK or USA

# Transform data to numeric values
# To use decision tree all data must be numeric.
# map() function - transforms text fields into numeric categories

comedians['Nationality'] = comedians['Nationality'].map({'UK':0, 'USA':1, 'N':2})
comedians['Go'] = comedians['Go'].map({'YES':1, 'NO':0})
comedians.head()

# The **Go** attribute contains the class label. In this case, go to the concert (**1 - yes**) or not go (**2 - no**).
# Split the dataset into two: data with characteristics and data with labels."

attributes_names = ['Age', 'Experience', 'Rank', 'Nationality']
attributes = comedians[attributes_names]
classes = comedians['Go']

# Build decision tree
tree = DecisionTreeClassifier()
tree = tree.fit(attributes.values,classes.values)

# Trained model
plt.figure(figsize=(15,8))
plot_tree(tree, filled = True, class_names=['No', 'Yes'], feature_names=attributes_names) # plot the image of the generated tree
plt.show()

# The tree shows, from trained data, how decisions will be made.
# At each node, there is the following information:
# The attribute name and its value to true or false.
# gigi: the method used to split the samples. The value varies between 0.0 (all samples have the same result, so there is no further division) and 0.5 (the maximum division done in the middle of the sample).
# Samples: number of samples that reached that node.
# Value: How many objects go to the next branches of the tree.

# In this case the most important attribute is Rank.
# If the rank is <= 6.5 the decision is not to go to the show. Otherwise it continues checking the next attributes.

# Prediction
# New object to test the model
print(tree.predict([[40,10,7,1]])) #age, experience, rank, language

# 0: The decision is not to go to the show

print(tree.predict([[80,30,4,0]])) #age, experience, rank, language

# 0: The decision is not to go to the show

print(tree.predict([[40,13,7,1]])) #age, experience, rank, language

# 1: The decision is to go to the show

"""# Evaluating the accuracy of the classification

Let's use decision tree to classify a database on wine quality. Then let's analyze the ranking performance.
"""

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
wines = pd.read_csv(url, sep =';')
wines.head(20)

wines['quality'].unique() #unique values

# Separate attributes from class characteristics

wines_columns = wines.columns[0:11].values.tolist() #12 firts columns in list. 13 is the target
wines_attributes = wines[wines_columns] 
wines_classes = wines['quality'] #target

# Split between training and testing. 10% of data for training

training_attributes, test_attributes, training_wines_classes, test_wines_classes = train_test_split(wines_attributes, wines_classes, test_size=0.1, random_state=10) #test: 10%

# Tree model
vine = DecisionTreeClassifier()
vine = vine.fit(training_attributes, training_wines_classes)

# Prediction of test data and verify accuracy

prediction_wines_classes = vine.predict(test_attributes)
accuracy = accuracy_score(test_wines_classes, prediction_wines_classes)
print('Classification Accuracy: {}'.format(accuracy))

# The accuracy is 62,50%. Reasonable.

# Tree image

plt.figure(figsize=(20,12))
plot_tree(vine, filled=True, rounded= True, class_names=['3','4','5','6','7','8'], feature_names=wines_columns)
plt.show()