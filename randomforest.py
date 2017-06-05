import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph. I like it most for plot
from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.cross_validation import KFold # use for cross validation
from sklearn.model_selection import GridSearchCV# for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm # for Support Vector Machine
from sklearn import metrics # for the check the error and accuracy of the model

data = pd.read_csv('D:/Anaconda3/envs/HR/project/breast_cancer_project/input/shuffled_data_del.csv')
data.sample()


features_mean= list(data.columns[0:2])
print(features_mean)

train, test = train_test_split(data, test_size = 0.3)# in this our main data is splitted into train and test
print(train.shape)
print(test.shape)

count = 0
a = test.diagnosis
for i in a:
    if i==0:
        count += 1
print(count)


prediction_var = features_mean
train_X = train[prediction_var]# taking the training data input
train_y=train.diagnosis# This is output of our training data
# same we have to do for test
test_X= test[prediction_var] # taking test data inputs
test_y =test.diagnosis   #output value of test dat

model=RandomForestClassifier(n_estimators=20)# a simple random forest model
model.fit(train_X,train_y)# now fit our model for traiing data

prediction=model.predict(test_X)# predict for the test data
# prediction will contain the predicted value by our model predicted values of dignosis column for test inputs
metrics.accuracy_score(prediction,test_y) # to check the accuracy


df_confusion = pd.crosstab(test_y, prediction, rownames=['Actual'], colnames=['Predicted'], margins=True)
print('\n')
print(df_confusion)
print('\n')

#Accuracy, precision, recall
df_confusion.as_matrix()
accuracy = (df_confusion[0][0] + df_confusion[1][1]) / (df_confusion[0][0] + df_confusion[0][1] + df_confusion[1][0] + df_confusion[1][1])
precision = (df_confusion[1][1]) / (df_confusion[1][1] + df_confusion[1][0])
recall = (df_confusion[1][1]) / (df_confusion[1][1] + df_confusion[0][1])
print('Accuracy : ' + str(accuracy))
print('Precision : ' + str(precision))
print('Recall : ' + str(recall))
print('\n')


featimp = pd.Series(model.feature_importances_, index=features_mean).sort_values(ascending=False)
print(featimp)

