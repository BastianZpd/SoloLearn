#Este es un archivo sin ningún tipo de lógica, sólo busca anotar las variables y funciones que voy aprendiendo
#en SoloLearn, para tener un respaldo y una guía de donde puedo acudir sobre lo que he aprendido.
#%%

from dataclasses import make_dataclass
import imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import column_or_1d

data = [15, 16, 18, 19, 22, 24, 29, 30, 34]

print("mean:", np.mean(data))
print("median:", np.median(data))
print("50th percentile (median=:", np.percentile(data, 50))
print("25th percentile:", np.percentile(data, 25))
print("75th percentile:", np.percentile(data, 75))
print("standard deviation:", np.std(data))
print("variance:", np.var(data))

#DataFrame(df) = table of data
#Head = first 5 rows of data
import pandas as pd
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
print(df.head())

#Describe = describe a table of statistics about the columns
import pandas as pd
pd.options.display.max_columns = 6
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
print(df.describe())

#To select a specific data or column, we select the column, in what is called a "Pandas Series"
#A series is like a DataFrame, but just a single column
import pandas as pd
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
col = df['Fare']
print(col)

#We can also select multiple columns from our original DataFrame, creating a smaller DataFrame
#We put these values in a list as follows: ['Age', 'Sex', 'Survived']
import pandas as pd
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
small_df = df[['Age', 'Sex', 'Survived']]
print(small_df.head())

#We often want our data in a different format than it originally comes in, for example,
#we can change the data in the column 'Sex' from 'Male' or 'Female' to boolean values.
#We can easily create a new column in our DataFrame
import pandas as pd
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['Male'] = df['Sex'] == 'male'
print(df.head())
#Or get if the passenger is in PClass or not
import pandas as pd
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['First Class'] = df['Pclass'] == 1
print(df.head())

#The lists or tables of data are often called a numpy array, we often will take the data from
#our pandas DataFrame and put it in numpy arrays, because a DataFrame is not the ideal format
#for doing calculations. Lets convert the Fare column to a numpy array
import pandas as pd
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
#We recall the Fare 'Series', and then use the '.values' attribute to get the values as numpy array
print(df['Fare'].values)

#If we have a DataFrame, we can still use the values attribute, but it returns a 2-dimensional numpy array
import pandas as pd
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
print(df[['Pclass', 'Fare', 'Age']].values)

#We use the numpy 'shape' attribute to determine the size of our numpy array
import pandas as pd
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
arr = df[['Pclass', 'Fare', 'Age']].values
print(arr.shape)
#This shows the number of rows and columns (rows,columns)

#If we have an array, we can select a single element from a numpy array with arr[0,1] = 1st row, 2nd column
#We can also select a single row print(arr[0]), that is the whole row of the first passenger
#To select a single column we have to use some special syntax: print(arr[:,2])
import pandas as pd
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
arr = df[['Pclass', 'Fare', 'Age']].values
print(arr[0,1])
print(arr[0])
print(arr[:,2])

#Masking
#Often times we want to select all the rows that meet a certain criteria
#We create what we call a mask first, This is an array of boolean values, of whether the passenger is a child or not
import pandas as pd
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
#First 10 values for simplicity
arr = df[['Pclass', 'Fare', 'Age']].values[:10]
mask = arr[:, 2] < 18
#We can make an array of a 'Filter', or as in this case, an array of the mask.
print(arr[mask])
#Generally we don't need to define the mask variable and can do te above in just a single line:
print(arr[arr[:, 2] < 18])
#Subset the array to get just the passengers in Pclass1
print(arr[arr[:, 0] == 1])
#Lets say we want to know how many of our passengers are children. Recall that True values are interpreted as 1
#and False values are interpreted as 0, so we can just sum up the array and that's equivalent to counting
#the number of True values.
print(mask.sum())
print((arr[:, 2] < 18).sum())
#Counting the number of passengers in Pclass 1
print((arr[:, 0] == 1).sum())
#mask1 = arr[:, 0] == 1
#print(mask1.sum())

#We can use the matplotlib library to plot our data.
#Plotting the data can often help us build intuition about our data. We use the scatter function to plot our data.
#The first argument of the scatter function is the x-axis, and then the y-axis.
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
arr = df[['Pclass', 'Fare', 'Age']].values
plt.scatter(df['Age'], df['Fare'])
#To make it easier to inrerpret, we can add x and y labels.
plt.xlabel('Age')
plt.ylabel('Fare')
#We can also use our data to coor code our scatter plot. This will give each of the 3 classes a different color
#We add the c parameter and give it a Pandas series. In this case our Pandas series has 3 possible values
#(1st, 2nd and 3rd class), so we'll see our datapoints each get one of three colors
plt.scatter(df['Age'], df['Fare'], c=df['Pclass'])

#Line
#The plot function does draw a line to approximately separate the 1st class from the 2nd and 3rd class
#From eyeballing, we'll put the line from (0,85) to (80,5)
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
arr = df[['Pclass', 'Fare', 'Age']].values
plt.scatter(df['Age'], df['Fare'], c=df['Pclass'])
plt.plot([0, 80], [85, 5])
#This ise the manually way, but there is a better way for doing this algorithmically
#In matplotlib we use the scatter function to create a scatter plot, and the plot function for a line

#In supervised Learning we call the label or thing we're trying to predict, the target. In unsupervised learning
#there is no known target. Within supervised learning, there is Classification and Regression. Classification problems
#are where the target is a categorical value (often True or False). Regression problems are where the target is a numerical value.
#For example, predicting housing prices is a regression problem. It's supervised, since we have historical data of the
#sales of houses in the past. It's regression, because the housing price is a numerical value.
#Predicting if someone will default on their loan is a classification problem. Again, it's supervised, since we
#have the historical data of whether past leanees defaulted, and it's a classification problem because we are trying to
#predict if the loan is in one of two categories (default or not)
#Loogistic Regression, while it has regression in its name is algorithm for solving classification problems, not regression problems.
#The Survived column in our DataFrame of the Titanic data we're trying to predict, it is called the target.
#There is a list of 1's and 0's that means if a passenger survived or not. The remaining columns are the information about the
#passenger that we can use to predict the target. We call each of these columns a feature.
#Features are the data we use to make our prediction. While we know whether each passenger in the dataset survived,
#we'd like to be able to make predictions about additional passengers that we weren't able to collect that data for.
#we'll build a machine learning model to help us do this. Sometimes, features are called predictors.

# #Line equation
# 0 = ax + by + c
# 0 = ((2)x + (1)y - 5); (2,1) (0,5); (2x2) + (1x1) - 5 = 0; (2x0) + (5x1) - 5 = 0;
# 0 = (1)x + (-1)y - 30
# #If we take a passenger's data, we can use this equation to determine which side of the line they fall on.
# #For example, let's say we have a passenger whose Fare is 100 and Age is 20.
# (1)100 + (-1)20 - 30 = 100 - 20 - 30 = 50;
# #Since its positive, the point is on the right side of the line and we'd predict that the passenger survived.
# #Now whit a passenger with a Fare of 10 and Age 50.
# (1)10 + (-1)50 - 30 = -70;

#Since this value is negative, the pint is on the left side of the line and we'd predict that the passenger didn't survive.
#There is many ways to find lines, but Logistic Regression is a way of mathematically finding the best line.
#In order to determine the best possible line to split our data, we need to have a way of scoring the line.
#First, let's look at a single datapoint. Ideally, if the datapoint is a passenger who survived, it would be on the right side
#of the line and far from the line if it's a datapoint for a passenger who didn't survive, it would be far from the line to the left.
#The further it is from the line, the more confident we are that it's on the correct side of the line.
#For each datapoint, we'll have a score that's a value between ' and 1. We can think of it as the probability that the passenger survives.
#If the value is close to 0 that point would be far to the left of the line and that means we're confident the passenger didn't survive.
#If it's close to 1 would be far to the right of the line and means we're confident the passenger did survive.
#A 0.5 value means the point falls directly on the line and we are uncertain if the passenger survives.
#The equation for calculating this score is:
# 1 / 1 + e ^ -(1x + by + c), This function is called the sigmoid.
#Though the intuition for it is far more important that the actual equation.
#Recall that the equation for the line is in the form
# 0 = ax + by + c (x is the Fare, y is the Age, and a, b & c are the coefficients that we control)
#The number e is the mathematical constant, approximately 2.71828
#Logistic Regression gives not just a prediction, but a probability (80%)
#To calculate how good our line is, we need to score whether our predictions are correct. Ideally if we predict with a high probability
#that a passenger survives (meaning the datapoint is far to the right of the line), then that passenger actually survives.
#So we'll get rewarded when we predict something correctly and penalized if we predict something incorrectly.
#Here is the likelihood equation. The intuition is more important than the equation.
#Likelihood = p if passenger survived, 1-p if passenger didn't survive
#The likelihood will be a value between 0 and 1. The higher the value, the better our line is.

#Let's look at a couple possibilities:
#- If the predicted probability p is 0.25 and the passenger didn't survive, we get a score of 0.75 (good)
#- If the predicted probability p is 0.25 and the passenger survived, we get a score of 0.25 (bad)

#We multiply all the individual scores for each datapoint together to get a score for our line. Thus we can compare
#different lines to determine the best one.
#Let's say for ease computation that we have 4 datapoints.
# 0.25, 0.75, 0.6, 0.8.
#We get the total score by multiplying the four scores together:
# 0.25 * 0.75 * 0.6 * 0.8 = 0.09
#The value is always going to be really small since it is the likelihood that our model predicts everything perfectly.
#A perfect model would have a predicted probability of 1 for all positive cases and 0 for all negative cases.
#The likelihood is how we score and compare possible choices of a best fit line.

#All of the basic machine learning algorithms are implemented in scikit-learn (sklearn)
#Before using sklearn to build a model, we need to prep the data with Pandas. First we need to make all our columns numerical.
#Recall to create the boolean column for Sex
df['Male'] = df['Sex'] == 'male'

#Now we create a numpy array called X with all the features. First select all the columns we are interested in and then use the values
#method to convert it to a numpy array
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children,' 'Fare']].values
#Now we take the target (Survived) and store it in a variable Y
Y = df['Survived'].values

import pandas as pd
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['Male'] = df['Sex'] == 'male'
X = df[['Pclass', 'Male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
Y = df['Survived'].values
print(X)
print(Y)

#It's standard practice to call our 2d array of features X and our 1d array of target values Y

#We start by importing the Logistic Regression model:
from sklearn.linear_model import LogisticRegression
#All sklearn models are built as Python classes. We first instantiate the class.
model = LogisticRegression()
#Now we can use our data that we previously prepared to train the model. The fit method is used for building the model.
#It takes two arguments: X (the features as a 2d numpy array) and Y (The target as a 1d numpy array)
#For simplicity let's first assume that we're building a Logistic Regression model using just the Fare and Age columns.
#First we define X to be the feature matrix and Y the target array.
X = df[['Fare', 'Age']].values
Y = df['Survived'].values
#Now we use the fit method to build the model
model.fit(X,Y)
#Fitting the model means using the data to choose a line of best fit. We can see the
#coefficients with the coef_ and intercept_attributes.
print(model.coef_, model.intercept_)

import pandas as pd
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
X = df[['Fare', 'Age']].values
Y = df['Survived'].values

model = LogisticRegression()
model.fit(X,Y)

print(model.coef_, model.intercept_)

#Now let's rebuild the model with all of the features

import pandas as pd
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['Male'] = df['Sex'] == 'male'
X = df[['Pclass', 'Male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
Y = df['Survived'].values

model = LogisticRegression()
model.fit(X,Y)

#Now we can use the predict method to make predictions.
model.predict(X)
#Print the first 5 rows
print(model.predict([[3, True, 22.0, 1, 0, 7.25]]))
print(model.predict(X[:5]))
print(Y[:5])

#We can get a sense of how good our model is by counting the number of datapoints it predicts correctly.
#This is called the accuracy score. Let's create an array that has the predicted Y values.
y_pred = model.predict(X)
#Now we create an array of boolean values of whether or not our model predicted each passenger correctly.
Y == y_pred
#To get the number of these that are true, we can use the numpy sum method.
print((Y == y_pred).sum())
# 714
# This means that of the 887 datapoints, the model makes the correct prediction for 714 of them.
# To get the percent
Y.shape[0] 
print((Y == y_pred).sum() / Y.shape[0])
# 0.8038
# Thus the model's accuracy is 80%. In other words, the model makes the correct prediction on 80% of the datapoints.
# This is a common enough calculation, that sklearn has already implemented it for us. So we can get the same result by using
# the score method. The score method uses the model to make a prediction for X and counts what percent of them match Y
print(model.score(X, Y))

import pandas as pd
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['Male'] = df['Sex'] == 'male'
X = df[['Pclass', 'Male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
Y = df['Survived'].values

model = LogisticRegression()
model.fit(X,Y)

y_pred = model.predict(X)
print((Y == y_pred).sum())
print((Y == y_pred).sum() / Y.shape[0])
print(model.score(X, Y))

#---------------------------------------------------------------------------------------------------------------------------------------

# Breast Cancer Dataset
# Now we'll build a Logistic Regression for a classification dataset.
# In the breast cancer dataset, each datapoint has measurements from an image of a breast mass and whether or not it's cancerous.
# The goal will be to use these measurements to predict if the mass is cancerous.
# This dataset is built right into scikit-learn so we won't need to read in a csv

from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()

# The object returned is an object similar to a Python dictionary. We can see the available keys with the keys method.
print(cancer_data.keys())
# We'll start by looking at DESCR, which gives a detailed description of the dataset.
print(cancer_data['DESCR'])

import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()
print(cancer_data.keys())
print(cancer_data['DESCR'])

# In the breast cancer dataset, there are several features that are calculated based on other columns. The process of figuring out
# what additional features to calculate is feature engineering

# Loading the Data into Pandas.
# The feature data is stored with the 'data' key. It's a numpy array with 569 rows and 30 columns.
# Recall to use shape to view data
cancer_data['data'].shape
# In order to put this in a Pandas DataFrame and make it readable, we want the column names. These are stored with the 'feature_names' key.
cancer_data['feature_names']
# Now we create a Pandas DataFrame
df = pd.DataFrame(cancer_data['data'],
columns = cancer_data['feature_names'])
print(df.head())

# In order to interpret malign or benign data (0's or 1's) we need to put the data in a DataFrame. This is given by target_names
df['target'] = cancer_data['target']

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

cancer_data = load_breast_cancer()

df = pd.DataFrame(cancer_data['data'],
columns = cancer_data['feature_names'])
df['target'] = cancer_data['target']
print(df.head())

# Now that we've taken a look at our data and gotten it into a comfortable format, we can build our feature matrix X
# and target array Y so that we can build a Logistic Regression model.

X = df[cancer_data.feature_names].values
Y = df['target'].values

# Now we create a Logistic Regression object and use the fit method to build the model

model = LogisticRegression()
model.fit(X, Y)

# When we run this code we get a Convergence Warning. this means that the model needs more time to find the optimal solution.
# One option is to increase the number of iterations. You can also switch to a different solver, which is what we will do.
# The solver is the algorithm that the model uses to find the equation of the line. You can see the possible solvers in the 
# Logistic Regression documentation.

model = LogisticRegression(solver = 'liblinear')
model.fit(X,Y)

# Let's see what the model predicts for the first datapoint in our dataset.
# Recall that the predict method takes a 2-dimensional array so we must put the
# datapoint in a list.

print("Prediction for datapoint 0: ")
model.predict([X[0]])

# So the model predicts that the first datapoint is benign.
# To see how well the model performs over the whole dataset, we use the score method
# to see the accuracy of the model.
print(model.score(X, Y))


# -------------------------------------------------------------------------------
# Model evaluation
# Accuracy: In the previous module, we calulcated how well our model performed using accuracy.
# Accuracy is  the percent of predictions that are correct.
# If you have 100 datapoints and predict 70 of them correctly and 30 incorrectly, 
# the accuracy is 70%
# Accuracy is a very straightforward and easy to understand metric, however it's
# not always the best one. For example, let's say i have a model to predict whether a credit
# card cahrge is fraudulent. Of 10000 credit card cahrds, we have 9900 legitimate charges and
# 100 fraudulent charges. I could build a model that just predicts that every single charge is
# legitimate and it would get 9900/10000(99%) of the predictions correct!
# Accuracy is a good measure if our classes are evenly split, but is very misleading if we
# have imbalanced calsses.

#Confusion Matrix
# As we noticed in the previous part, we care not only about how many datapoints we predict
# the correct class for, we care about how many of the positive datapoints we predict
# corrrectly for as well as how many of the negative datapoints we predict correctly.
# We can see all the important values in what is called the Confusion Matrix (or Error Matrix
# or Table of Confusion)
# The confusion matrix is a table showing four values:
# Datapoints we predicted positive that are actually positive
# Datapoints we predicted positive that are actually negative
# Datapoints we predicted negative that are actually positive
# Datapoints we predicted negative that are actually negative

# The first an fourth are the datapoins we predicted correctly and the second and third are 
# the datapoins we predicted incorrectly.

# In our Titanic dataset, we have 887 passengers, 342 survived (positive) and 545 didn't survive (negative). The model we built in the
# previous module has the following confusion matrix

# Predicted positive - Actual positive = 233
# Predicted positive - Actual negative = 65
# Predicted negative - Actual positive = 109
# Predicted negative - Actual negative = 480

# The PP-AP and PN-AN are the counts of the predictions that we got correct. So of the 342 passengers that survived, 
# we predicted 233 of them correctly (109 of them incorrectly). Of the 545 passengers that didn't sutvive, we predicted 
# 480 correctly (65 incorrectly)
# We can use the confusion matrix to compute the accuracy. As a reminder, the accuracy is the number of datapoints predicted 
# correctly divided by the total number of datapoints.

# (233+480)/(233+65+109+480) = 713/887 = 80.38%

# This is indeed the same value we got in the previous module.
# The confusion matrix fully describes how a model performs on a dataset, though is difficult to use to compare models.

# True positives, true negatives, false positives and false negatives.
# We have names for each square of the confusion matrix 
# True positive (TP), true negative (TN), false positive (FP), false negative (FN)
# The terms can be a little hard to keep track of. The way to remember is that the second word is what our prediction is 
# (positive or negative) and the first word is whether that prediction was correct (true or false).
# you'll often see the confusion matrix described as follows:
# TP - FP
# FN - TN

# Two commonly used metrics for classifitacion are precision and recall. conceptually, precision refers to the percentage of positive results 
# which are relevant and recall to the percentage of positive cases correctly classified.
# Both can be defined using wuadrants from the confusion matrix

# Precision is the percent of the model's positive predictions that are correct. We define it as follows:
# precision = # positives predicted correctly / # positive predictions = TP/TP+FP

# If we look at our confusion matrix for our model for the Titanic dataset, we can calculate the precision. 
# Precision = 233 / (233 + 65) = 0.7819

# Recall is the percent of positive cases that the model prdicts correctly. Again, we will be using the confusion matrix to compute our result
# Here we mathematically define the recall:
# recall = # positives predicted correctly / #positive cases = TP/TP+FN

# Let's calculate the recall for our model for the Titanic dataset.
# recall = 233/233+109 = 0.6813
# Recall is a measure of how many of the positive cases the model can recall 

# Precision & Recall Trade-off 
# We often will be in a situation of choosing between increasing the recall (while lowering the precision) or increasing the precision 
# (and lowering the recall). It will depend on the situation which we'll want to maximize. 

# For example, let's say we're building a model to predict if a credict card charge is fraudulent. The positive cases for our model
# are fraudulent charges and the negative cases are legitimate charges.
# Let's consider two scenarios:
# 1. If we predict the charge is fraudulent, we'll reject the charge
# 2. If we predict the charge is fraudulent, we'll call the customer to confirm the charge.
# In case 1, it's a huge inconvenience for the customer when the model predicts fraud incorrectly (a false positive). In case4 2, a false positive
# is a minor inconvenience for the costumer.
# The higher the false positives, the lower the precision. Because of the high cost to false positives in the first case, it would be 
# worth having a low recall in order to have a ver high precision. In case 2, you would want more of a balance between precision and recall.
# There's no hard and fast rule on what values of precision and recall you're shooting for. It always depends on the dataset and the application. 

# Accuracy was an appealing metric because it was a single number. Precision and recall are two numbers so it's not always
#  obvious how to choose between two models if one has a highet precision and the other has a higher recall. The F1 score is
#  an average of precision and ercall so that we have a single score for our model
#  Here's the mathematical formula for the F1 score
#  F1 = 2 * precision*recall / precision+recall 
#  Let's calculate the F1 score for our model for the Titanic dataset.
#  We'll use the precision and recall numbers that we previously calculated. the precision is 0.7819 and the recall is 0.6813
#  2*(o.7819)*(0.6813)/&(0.7819+0.6813) = 0.7281
#  The F1 score is the harmonic mean of the precision and recall values.

#  Accuracy, precision, recall & F1 score in sklearn
# Scikit-learn has a function built in for each of the metrics that we have introduced. 
# We have a separate function for each of the accuracy, precision, recall and F1 score. 

# In order to use them, let's start by recalling our code from the previous module to build a Logistic Regression model.
# The code reads in the Titanitc dataset from the csv file and puts it in a Pandas DataFrame. Then we create a feature 
# matrix X and target values Y. we create a Logistic Regression model and fit it to our dataset.
# Finally, we create a variable y_pred of our predictions

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['Male'] = df['Sex'] == 'male'
X = df[['Pclass', 'Male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
model = LogisticRegression()
model.fit(X,Y)

y_pred = model.predict(X)

# Now we're ready to use our metric functions. Let's import them from scikit-learn 

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve

# Each function takes two 1-dimensional numpy arrays: The true values of the target & the predicted values of the target 
# We have the true values of the target and the predicted values of the target. Thus we can use the metric functions as follows 

print("Accuracy: ", accuracy_score(y, y_pred))
print("Precision: ", precision_score(y, y_pred))
print("Recall: ", recall_score(y, y_pred))
print("F1: ", f1_score(y, y_pred))

# We see that the accuracy is 80% which means that 80% of the model's predictions are correct.
# The precision is 78%, which we recall is the percent of the model's positive predictions that are correct. 
# The recall is 68%, which is the percent of the positive cases that the model predicted correctly. 
# The F1 score is 73%, which is an average of the precision and recall. 

# With a single model, the metric values do not tel us a lot. For some problems a value of 60% is good,
# and for others a value of 90% is good, depending on the difficulty of the problem. 
# We will use the metric values to compare different models to pick the best one.

# Scikit-learn has a confusion matrix function that we can use to get the four values in the confusion matrix (truse positives,
# false positives, false negatives and true negatives). Assuming y is our true target calues and y_pred is the predicted vales,
# we can use the confusion_matrix as follows:

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, y_pred))

# Scikit-learn reverses the confusion matrix to show the negative counts first, since negative values correspond to 0
# and positive to 1, scikit-learn has ordered them in this order. Make sure you double check that you are interpreting the values corectly

# Overfitting
# So far we've built a model with all of our data and then seen how well it performed on the same data. This is
# artificially intlating our numbers since our model, in efect, got to see the answers to the quiz before we 
# gave it the quiz. this can lead to what we call overfitting. Overfitting is when we perform well on the data
# the model has already seen, but we don't perform well on new data.

# We can visually see an overfit model as follows. The line is too closely trying to get every single datapoint on the correct
# side of the line but is missing the essence of the data. 
# The more features we have in our dataset, the more prone we'll be to overfitting

# Training Set and Test Set 
# To give a model a fair assessment, we'dl ike to know how well our data would perform on data it hasn't seen yet.
# In action, our model will be making predictions on data we don't know the answer to, so we'd like to evaluate how wel our
# model does on new data, not just the data it's already seen. To simulate making predictions on new  unseen data, we can break
# our dataset into a training set anda test set. The training set is used for building the models. 
# The test set is used for evaluating the models. We split our data before building the model, thuis the model has no knowledge
# of the test set and we'll be giving it a fair assessment.
# If our dataset has 200 datapoint in it, breaking it into a training set and test set might look like 150 datapoints
# for training set and 50 datapoints to test set 
# A Standard breakdown is to put 70-80% of our data in the training set and 20-30% in the test set. Using less data in
# the training set means that our model won't have as much data to learn from, so we want to give it as much as possible while
# still leaving enough for evaluation.

# Training and Testing in Sklearn 
# Scikit-learn has a function built in for splitting the data into a training set and a test set 
# Assuming we have a 2-dimensional numpy array  of our features and a 1-dimensional numpy array y of the target, we can use
# the train_test_split function. It will randomly put each datapoint in either the training set
# or the test set. By default the training set is 75% of the data and the test set is the remaining 25% of the data 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

# Let's use the shape attribute to see the sizes of our datasets
print("Whole dataset: ", X.shape, y.shape)
print("Training set: ", X_train.shape , y_test.shape)


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['Male'] = df['Sex'] == 'male'
X = df[['Pclass', 'Male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X,y)
print("Whole dataset: ", X.shape, y.shape)
print("Training set: ", X_train.shape , y_train.shape)
print("Test dataset: ", X_test.shape, y_test.shape)

# We can see that of the 887 datapoints in our dataset, 665 of them are in our training set and 222 are in the test set 
# Every datapoint from our dataset is used exactly once, either in the training set or the test set 
# Note that we have 6 features in our datset, so we still have 6 features in botho ur training set and test set
# We can change the size of our training set by using the train_size parameter. E.g. train_test_split(X,y,train_size=0.6)
# would put 60% of the data in the training set and 40% in the test set 

# Now that we knot how to split our data into a training set and a test set, we need to modigy how we build 
# and evaluate the model. All of the model building is done with the training set and all of the evaluation is done with the test set 
# In the last module, we built a model and evaluated it on the same datset. Now we build the model using the training set 

model = LogisticRegression()
model.fit(X_train, y_train)
# And we evaluate the model using the test set
print(model.score(X_test, y_test))

# In fact, all of the metrics we calculate in the previous parts should be calculated on the test set 
y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))

# Our accuracy, precision, recall and F1 score values are actually very similar to the values when we used the entire dataset 
# This is a sign our model is not overfit


# Logistic Regression Threshold 
# If you recall in Lesson 2, we talked about the trade-off between precision and recall. With a Logistic Regression mode, we have an 
# easy way of shifting between emphazising precision and emphasizing recall. 
# The Logistic Regression model doesn't just return a prediction, but it returns a probability value between 0 and 1. Tipically, we say if the
# value is >=0.5, we predict the passenger survived, and if the value is <0.5, the passenger didn-t survive. However, we could choose any threshold 
# between 0 and 1. 
# If we make the threshold higher, we'll have fewer positive predictions, but our positive predictions are more likely to be correct. This means 
# that the precision would be higher and the recall lower. On the other hand, if we make the threshold lower, we'll have more positive predictions, 
# so we're more likely to catch all the positive cases. This means that the recall would be higher and the precision lower.
# --Each choice of a threshold is a diferent model. An ROC (Receiver operating characteristic) Curve is a graph showing 
# all of the possible models and their performance--

# Sensitivity & Specificity

# An ROC Curve is a graph of the sensitivity vs. the specificity. These values demonstrate the same trade-off that precision and recall demonstrate.
# Let’s look back at the Confusion Matrix, as we’ll be using it to define sensitivity and specificity.

# Sensitivity = recall = #positives predicted correctly / #positive cases = TP/TP+FN
# The specificity is the true negative rate. It's calculated as follows
# Specificity = #negatives predicted correctly / #negative cases = TN/TN+FP

# We’ve done a train test split on our Titanic dataset and gotten the following confusion matrix. We have 96 positive cases and 126 negative cases in our test set.
# TP = 61
# FP = 21
# FN = 35
# TN = 105

# Let’s calculate the sensitivity and specificity.
# Sensitivity = 61/96 = 0.6354
# Specificity = 105/126 = 0.8333

# The goal is to maximize these two values, though generally making one larger makes the other lower. It will depend on the situation whether 
# we put more emphasis on sensitivity or specificity.
# While we generally look at precision and recall values, for graphing the standard is to use the sensitivity and specificity. 
# It is possible to build a precision-recall curve, but this isn’t commonly done.


# Sensitivity & Specificity in Scikit-learn
# Scikit-learn has not defined functions for sensitivity and specificity, but we can do it ourselves. Sensitivity is the same as recall, so it is easy to define.
from sklearn.metrics import recall_score
sensitivity_score = recall_score
print(sensitivity_score(y_test, y_pred)) 
# 0.6829268292682927

# Now, to define specificity, if we realize that it is also the recall of the negative class, we can get the value from the sklearn function 
# precision_recall_fscore_support.

# Let’s look at the output of precision_recall_fscore_support.
from sklearn.metrics import precision_recall_fscore_support
print(precision_recall_fscore_support(y, y_pred))

# The second array is the recall, so we can ignore the other three arrays. There are two values. The first is the recall of the negative class and 
# the second is the recall of the positive class. The second value is the standard recall or sensitivity value, and you can see the value matches 
# what we got above. The first value is the specificity. So let’s write a function to get just that value.
def specificity_score(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    return r[0]
print(specificity_score(y_test, y_pred)) 
# 0.9214285714285714

# Note that in the code sample we use a random state in the train test split so that every time you run the code you will get the same results.

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_recall_fscore_support

sensitivity_score = recall_score
def spedificity_score(y_true, y_pred):
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
        return r[0]

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['Male'] = df['Sex'] == 'male'
X = df[['Pclass', 'Male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Sensitivity: ", sensitivity_score(y_test, y_pred))
print("Specificity: ", specificity_score(y_test, y_pred))

# Sensitivity is the same as the recall (or recall of the positive class) 
# and specificity is the recall of the negative class.


# Adjusting the Logistic Regresison Threshold in sklearn
# When you use scikit-learn's predict methos, you are given 0 and 1 values of the prediction. However, behind the scenes the Logistic Regression
# model is getting a probability value between 0 and 1 for each datapoint and then reounding to either 0 or 1.
# If we want to choose a different threshold besides 0.5 we'll want those probability values. We can use the predict_proba function to get them.
# (model.predict_proba(X_test)
# The result is a numpy array with 2 values for each datapoint (e.g. [0.78, 0.22]). You’ll notice that the two values sum to 1. The first value is 
# the probability that the datapoint is in the 0 class (didn’t survive) and the second is the probability that the datapoint is in the 1 class 
# (survived). We only need the second column of this result, which we can pull with the following numpy syntax.
# model.predict_proba(X_test)[:, 1]
# Now we just want to compare these probability values with our threshold. Say we want a threshold of 0.75. We compare the above array to 0.75. 
# This will give us an array of True/False values which will be our array of predicted target values.
# y_pred = model.predict_proba(X_test)[:, 1] > 0.75
# A threshold of 0.75 means we need to be more confident in order to make a positive prediction. This results in fewer positive predictions and 
# more negative predictions.

# Now we can use any scikit-learn metrics from before using y_test as our true values and y_pred as our predicted values.
print("precision:", precision_score(y_test, y_pred))
print("recall:", recall_score(y_test, y_pred))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score

sensitivity_score = recall_score, precision_recall_fscore_support

sensitivity_score = recall_score
def spedificity_score(y_true, y_pred):
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
        return r[0]

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['Male'] = df['Sex'] == 'male'
X = df[['Pclass', 'Male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LogisticRegression()
model.fit(X_train, y_train)

print("Predicted proba: ")
print(model.predict_proba(X_test))

y_pred = model.predict_proba(X_test)[:, 1] > 0.75

print("Precision :", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))

# Setting the threshold to 0.5 we would get the original Logistic Regression model. Any other threshold value yields an alternative model.


# How to Build an ROC Curve

# The ROC curve is a graph of the specificity vs the sensitivity. We build a Logistic Regression model and then calculate the specificity 
# and sensitivity for every possible threshold. Every predicted probability is a threshold. If we have 5 datapoints with the following predicted 
# probabilities: 0.3, 0.4, 0.6, 0.7, 0.8, we would use each of those 5 values as a threshold.
# Note that we actually plot the sensitivity vs (1-specificity). There is no strong reason for doing it this way besides that it’s the standard.
# Let’s start by looking at the code to build the ROC curve. Scikit-learn has a roc_curve function we can use. The function takes the true 
# target values and the predicted probabilities from our model.
# We first use the predict_proba method on the model to get the probabilities. Then we call the roc_curve function. The roc_curve function returns
# an array of the false positive rates, an array of the true positive rates and the thresholds. The false positive rate is 1-specificity (x-axis)
# and the true positive rate is another term for the sensitivity (y-axis). The threshold values won’t be needed in the graph.
# Here’s the code for plotting the ROC curve in matplotlib. Note that we also have code for plotting a diagonal line. This can help us visually 
# see how far our model is from a model that predicts randomly.
# We assume that we already have a dataset that has been split into a training set and test set.

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1])

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1 - specificity')
plt.ylabel('sensitivity')
plt.show()

# As we don’t use the threshold values to build the graph, the graph does not tell us what threshold would yield each of the possible models.


# ROC Curve Interpretation

# The ROC curve is showing the performance, not of a single model, but of many models. Each choice of threshold is a different model.

# Each point A, B & C refers to a model with a different threshold.

# Model A has a sensitivity of 0.6 and a specificity of 0.9 (recall that the graph is showing 1-specificity).
# Model B has a sensitivity of 0.8 and a specificity of 0.7.
# Model C has a sensitivity of 0.9 and a specificity of 0.5.

# How to choose between these models will depend on the specifics of our situation.
# The closer the curve gets to the upper left corner, the better the performance. The line should never fall below the diagonal line 
# as that would mean it performs worse than a random model.

# Picking a Model from the ROC Curve

# When we’re ready to finalize our model, we have to choose a single threshold that we’ll use to make our predictions. 
# The ROC curve is a way of helping us choose the ideal threshold for our problem.

# Let’s again look at our ROC curve with three points highlighted:
# If we are in a situation where it’s more important that all of our positive predictions are correct than that we catch 
# all the positive cases (meaning that we predict most of the negative cases correctly), we should choose the model with higher
# specificity (model A).

# If we are in a situation where it’s important that we catch as many of the positive cases as possible, we should choose the model 
# with the higher sensitivity (model C).

# If we want a balance between sensitivity and specificity, we should choose model B.
# It can be tricky keeping track of all these terms. Even experts have to look them up again to ensure they are interpreting the values correctly.


# Area Under the Curve

# We’ll sometimes what to use the ROC curve to compare two different models. Here is a comparison of the ROC curves of two models.
# You can see that the blue curve outperforms the orange one since the blue line is almost always above the orange line.
# To get an empirical measure of this, we calculate the Area Under the Curve, also called the AUC. This is the area under the ROC curve. 
# It’s a value between 0 and 1, the higher the better.
# Since the ROC is a graph of all the different Logistic Regression models with different thresholds, the AUC does not measure the 
# performance of a single model. It gives a general 
# sense of how well the Logistic Regression model is performing. To get a single model, you still need to find the optimal threshold 
# for your problem.

# Let’s use scikit-learn to help us calculate the area under the curve. We can use the roc_auc_score function.
# (roc_auc_score(y_test, y_pred_proba[:,1]) 

# Here are the values for the two lines:
# Blue AUC: 0.8379
# Orange AUC: 0.7385
# You can see empirically that the blue is better.

# We can use the roc_auc_score function to calculate the AUC score of a Logistic Regression model on the Titanic dataset. 
# We build two Logistic Regression models, model1 with 6 features and model2 with just Pclass and male features. 
# We see that the AUC score of model1 is higher.

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['Male'] = df['Sex'] == 'male'
X = df[['Pclass', 'Male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y)

model1 = LogisticRegression()
model1.fit(X_train, y_train)
y_pred_proba1 = model1.predict_proba(X_test)
print("Model 1 AUC Score: ", roc_auc_score(y_test, y_pred_proba1[:, 1]))

model2 = LogisticRegression()
model2.fit(X_train[:, 0:2], y_train)
y_pred_proba2 = model2.predict_proba(X_test[:, 0:2])
print("Model 2 AUC Score: ", roc_auc_score(y_test, y_pred_proba2[:, 1]))


# Concerns with Training and Test Set

# We are doing evaluation because we want to get an accurate measure of  how well the model performs. 
# If our dataset is small, our test set is  going to be small. Thus it might not be a good random assortment of  
# datapoints and by random chance end up with easy or difficult datapoints in our evaluation set.

# Since our goal is to get the best possible measure of our metrics (accuracy, precision, recall and F1 score), 
# we can do a little better than just a single training and test set.

# As we can see, all the values in the training set are never used to evaluate. It would be unfair to build 
# the model with the training set and then evaluate with the training set, but we are not getting as full a 
# picture of the model performance as possible.

# To see this empirically, let’s try running the code from Lesson 3 which does a train/test split. 
# We’ll re-run it a few times and see the results. Each row is the result of a different random train/test split.

# You can see that each time we run it, we get different values for the metrics. The accuracy ranges 
# from 0.79 to 0.84, the precision from 0.75 to 0.81 and the recall from 0.63 to 0.75. These are wide ranges 
# that just depend on how lucky or unlucky we were in which datapoints ended up in the test set.

# Instead of doing a single train/test split, we’ll split our data into a training set and test set multiple times.


# Multiple Training and Test Sets

# We learned in the previous part that depending on our test set, we can get different values for the evaluation metrics. 
# We want to get a measure of how well our model does in general, not just a measure of how well it does on one specific test set.
# Instead of just taking a chunk of the data as the test set, let’s break our dataset into 5 chunks. 
# Let’s assume we have 200 datapoints in our dataset.
# Each of these 5 chunks will serve as a test set. When Chunk 1 is the test set, we use the remaining 4 chunks as the training set. 
# Thus we have 5 training and test sets as follows.

# 1. Train-Train-Train-Train-Test 
# 2. Train-Train-Train-Test-Train 
# 3. Train-Train-Test-Train-Train 
# 4. Train-Test-Train-Train-Train 
# 5. Test-Train-Train-Train-Train 

# Each of the 5 times we have a test set of 20% (40 datapoints) and a training set of 80% (160 datapoints).
# Every datapoint is in exactly 1 test set.


# Building and Evaluating with Multiple Training and Test Sets

# In the previous part we saw how we could make 5 test sets, each with a different training set.
# Now, for each training set, we build a model and evaluate it using the associated test set. Thus we build 5 models and calculate 5 scores.
# Let’s say we are trying to calculate the accuracy score for our model.

# 1. Train-Train-Train-Train-Test ACC = 0.83
# 2. Train-Train-Train-Test-Train ACC = 0.79
# 3. Train-Train-Test-Train-Train ACC = 0.78
# 4. Train-Test-Train-Train-Train ACC = 0.80
# 5. Test-Train-Train-Train-Train ACC = 0.75

# We report the accuracy as the mean of the 5 values:
# (0.83+0.79+0.78+0.80+0.75)/5 = 0.79
# If we had just done a single training and test set and had randomly gotten the first one, we would have reported an accuracy of 0.83. 
# If we had randomly gotten the last one, we would have reported an accuracy of 0.75. Averaging all these possible values helps eliminate 
# the impact of which test set a datapoint lands in.

# You will only see values this different when you have a small dataset. With large datasets we often just do a training and test set for simplicity.

# This process for creating multiple training and test sets is called k-fold cross validation. The k is the number of chunks we split our 
# dataset into. The standard number is 5, as we did in our example above.
# Our goal in cross validation is to get accurate measures for our metrics (accuracy, precision, recall). We are building extra models in 
# order to feel confident in the numbers we calculate and report.


# Final Model Choice in k-fold Cross Validation

# Now we have built 5 models instead of just one. How do we decide on a single model to use?
# These 5 models were built just for evaluation purposes, so that we can report the metric values. We don’t actually need these models 
# and want to build the best possible model. The best possible model is going to be a model that uses all of the data. So we keep track of 
# our calculated values for our evaluation metrics and then build a model using all of the data.
# This may seem incredibly wasteful, but computers have a lot of computation power, so it’s worth using a little extra to make sure we’re 
# reporting the right values for our evaluation metrics. We’ll be using these values to make decisions, 
# so calculating them correctly is very important.
# Computation power for building a model can be a concern when the dataset is large. In these cases, we just do a train test split.


# KFold Class

# Scikit-learn has already implemented the code to break the dataset into k chunks and create k training and test sets.

# For simplicity, let’s take a dataset with just 6 datapoints and 2 features and a 3-fold cross validation on the dataset. 
# We’ll take the first 6 rows from the Titanic dataset and use just the Age and Fare columns.

X = df[['Age', 'Fare']].values[:6]
y = df['Survived'].values[:6]

# We start by instantiating a KFold class object. It takes two parameters: n_splits (this is k, the number of chunks to create) 
# and shuffle (whether or not to randomize the order of the data). It’s generally good practice to shuffle the data since 
# you often get a dataset that’s in a sorted order.

kf = KFold(n_splits=3, shuffle=True)

# The KFold class has a split method that creates the 3 splits for our data.
# Let’s look at the output of the split method. The split method returns a generator, so we use the list function to turn it into a list.

list(kf.split(X))

from sklearn.model_selection import KFold
import pandas as pd

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
X = df[['Age', 'Fare']].values[:6]
y = df['Survived'].values[:6]

kf = KFold(n_splits=3, shuffle=True)
for train, test in kf.split(X):
    print(train, test)

# As we can see, we have 3 training and testing sets as expected. The first training set is made up of datapoints 0, 2, 3, 5 
# and the test set is made up of datapoints 1, 4.
# The split is done randomly, so expect to see different datapoints in the sets each time you run the code.


# Creating Training and Test Sets with the Folds

# We used the KFold class and split method to get the indices that are in each of the splits. 
# Now let’s use that result to get our first (of 3) train/test splits.
# First let’s pull out the first split.

splits = list(kf.split(X))
first_split = splits[0]
print(first_split)
# (array([0, 2, 3, 5]), array([1, 4]))

# The first array is the indices for the training set and the second is the indices for the test set. Let’s create these variables.
train_indices, test_indices = first_split
print("training set indices:", train_indices)
print("test set indices:", test_indices)
# training set indices: [0, 2, 3, 5]
# test set indices: [1, 4]

# Now we can create an X_train, y_train, X_test, and y_test based on these indices.
X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

# If we print each of these out, we’ll see that we have four of the datapoints in X_train and their target values in y_train. 
# The remaining 2 datapoints are in X_test and their target values in y_test.
print("X_train")
print(X_train)
print("y_train", y_train)
print("X_test")
print(X_test)
print("y_test", y_test)


from sklearn.model_selection import KFold
import pandas as pd

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
X = df[['Age', 'Fare']].values[:6]
y = df['Survived'].values[:6]

kf = KFold(n_splits=3, shuffle=True)
for train, test in kf.split(X):
    print(train, test)
splits = list(kf.split(X))
first_split = splits[0]
print(first_split)

train_indices, test_indices = first_split
print("training set indices:", train_indices)
print("test set indices:", test_indices)

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

print("X_train")
print(X_train)
print("y_train", y_train)
print("X_test")
print(X_test)
print("y_test", y_test)

# At this point, we have training and test sets in the same format as we did using the train_test_split function.

# Build a Model


# Now we can use the training and test sets to build a model and make a prediction like before. 
# Let’s go back to using the entire dataset (since 4 datapoints is not enough to build a decent model).
# Here’s the entirety of the code to build and score the model on the first fold of a 5-fold cross validation. 
# Note that the code for fitting and scoring the model is exactly the same as it was when we used the train_test_split function.

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import pandas as pd

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['Male'] = df['Sex'] == 'male'
X = df[['Pclass', 'Male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

kf = KFold(n_splits=5, shuffle=True)

splits = list(kf.split(X))
train_indices, test_indices = splits[0]

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

# So far, we’ve essentially done a single train/test split. 
# In order to do a k-fold cross validation, we need to do use each of the other 4 splits to build a model and score the model.


# Loop Over All the Folds

# We have been doing one fold at a time, but really we want to loop over all the folds to get all the values. 
# We will put the code from the previous part inside our for loop.

scores = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
print(scores)
# [0.75847, 0.83146, 0.85876, 0.76271, 0.74011]

# Since we have 5 folds, we get 5 accuracy values. Recall, to get a single final value, we need to take the mean of those values.
print(np.mean(scores))
# 0.79029

# Now that we’ve calculated the accuracy, we no longer need the 5 different models that we’ve built. For future use, we just want a single model. 
# To get the single best possible model, we build a model on the whole dataset. If we’re asked the accuracy of this model, we use the 
# accuracy calculated by cross validation (0.79029) even though we haven’t actually tested this particular model with a test set.

final_model = LogisticRegression()
final_model.fit(X, y)

# Expect to get slightly different values every time you run the code. The KFold class is randomly splitting up the data each time, 
# so a different split will result in different scores, though you should expect the average of the 5 scores to generally be about the same.


# Comparing Different Models

# So far we’ve used our evaluation techniques to get scores for a single model. These techniques will become incredibly useful 
# as we introduce more models and want to determine which one performs the best for a specific problem.

# Let’s use our techniques to compare three models:
# • A logistic regression model using all of the features in our dataset
# • A logistic regression model using just the Pclass, Age, and Sex columns
# • A logistic regression model using just the Fare and Age columns

# We wouldn’t expect the second or third model to do better since it has less information, but we might determine that using 
# just those two or three columns yields comparable performance to using all the columns.
# Evaluation techniques are essential for deciding between multiple model options.


# Building the Models with Scikit-learn

# Let’s write the code to build the two models in scikit-learn. Then we’ll use k-fold cross validation to calculate the accuracy, 
# precision, recall and F1 score for the two models so that we can compare them.

# First, we import the necessary modules and prep the data as we’ve done before.
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'

# Now we can build the KFold object. We’ll use 5 splits as that’s standard. Note that we want to create a single 
# KFold object that all of the models will use. It would be unfair if different models got a different split of the data.
kf = KFold(n_splits=5, shuffle=True)

# Now we’ll create three different feature matrices X1, X2 and X3. All will have the same target y.
X1 = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
X2 = df[['Pclass', 'male', 'Age']].values
X3 = df[['Fare', 'Age']].values
y = df['Survived'].values

# Since we’ll be doing it several times, let’s write a function to score the model. This function uses the KFold object 
# to calculate the accuracy, precision, recall and F1 score for a Logistic Regression model with the given feature matrix X and target array y.

# Then we call our function three times for each of our three feature matrices and see the results.

def score_model(X, y, kf): 
    accuracy_scores = [] 
    precision_scores = [] 
    recall_scores = [] 
    f1_scores = [] 
    for train_index, test_index in kf.split(X): 
        X_train, X_test = X[train_index], X[test_index] 
        y_train, y_test = y[train_index], y[test_index] 
    model = LogisticRegression() 
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test) 
    accuracy_scores.append(accuracy_score(y_test, y_pred)) 
    precision_scores.append(precision_score(y_test, y_pred)) 
    recall_scores.append(recall_score(y_test, y_pred)) 
    f1_scores.append(f1_score(y_test, y_pred)) 
    print("accuracy:", np.mean(accuracy_scores)) 
    print("precision:", np.mean(precision_scores)) 
    print("recall:", np.mean(recall_scores)) 
    print("f1 score:", np.mean(f1_scores)) 

print("Logistic Regression with all features")
score_model(X1, y, kf)
print()
print("Logistic Regression with Pclass, Sex & Age features ")
score_model(X2, y, kf)
print()
print("Logistic Regression with Fare &Age features ")
score_model(X3, y, kf)

# We’ll interpret these results in the next part.
# Expect to get slightly different results every time you run the code. The k-fold splits are chosen randomly, 
# so there will be a little variation depending on what split each datapoint ends up in.

# Logistic Regression with all features
# accuracy: 0.7959055418015616
# precision: 0.764272127669388
# recall: 0.6783206767486641
# f1 score: 0.7163036778464393

# Logistic Regression with Pclass, Sex & Age features
# accuracy: 0.7981908207960389
# precision: 0.7715749823848419
# recall: 0.6830371999703425
# f1 score: 0.7232930032930033

# Logistic Regression with Fare & Age features
# accuracy: 0.6538944962864216
# precision: 0.6519918328980114
# recall: 0.23722965720416847
# f1 score: 0.34438594236494796

# If we compare the first two models, they have almost identical scores. The third model has lower scores for all four metrics. 
# The first two are thus much better options than the third. This matches intuition since the third model doesn’t have access 
# to the sex of the passenger. Our expectation is that women are more likely to survive, so having the sex would be a very valuable predictor.

# Since the first two models have equivalent results, it makes sense to choose the simpler model, the one that uses the Pclass, Sex & Age features.

# Now that we’ve made a choice of a best model, we build a single final model using all of the data.
model = LogisticRegression()
model.fit(X1, y)

# Now we can make a prediction with our model.

model.predict([[3, False, 25, 0, 1, 2]])



# A Nonparametric Machine Learning Algorithm

# So far we’ve been dealing with Logistic Regression. In Logistic Regression, we look at the data graphically and draw a line to separate the data. 
# The model is defined by the coefficients that define the line. These coefficients are called parameters. Since the model is defined by these 
# parameters, Logistic Regression is a parametric machine learning algorithm.

# In this module, we’ll introduce Decision Trees, which are an example of a nonparametric machine learning algorithm. 
# Decision Trees won’t be defined by a list of parameters as we’ll see in the upcoming lessons.
# Every machine learning algorithm is either parametric or nonparametric.


# Tree Terminology

# The reason many people love decision trees is because they are very easy to interpret. It is basically a flow chart of questions that 
# you answer about a datapoint until you get to a prediction.

# Here’s an example of a Decision Tree for the Titanic dataset. We’ll see in the next lesson how this tree is constructed.

# Each of the rectangles is called a node. The nodes which have a feature to split on are called internal nodes. 
# The very first internal node at the top is called the root node. The final nodes where we make the predictions of survived/didn’t survive 
# are called leaf nodes. Internal nodes all have two nodes below them, which we call the node’s children

# The terms for trees (root, leaf) come from an actual tree, though it’s upside down since we generally draw the root at the top. 
# We also use terms that view the tree as a family tree (child node & parent node).

# Interpreting a Decision Tree


# To interpret this Decision Tree, let’s run through an example. Let’s say we want to know the prediction for a 10 year old 
# male passenger in Pclass 2. At the first node, since the passenger’s sex is male, we go to the right child. 
# Then, since their age 10 which is <=13 we go to the left child, and at the third node we go to the right child since the Pclass is 2. 
# In the following diagram we highlight the path for this passenger.

# Note that there are no rules that we use every feature, or what order we use the features, or for a continuous value (like Age), 
# where we do the split. It is standard in a Decision Tree to have each split just have 2 options.
# Decision Trees are often favored if you have a non-technical audience since they can easily interpret the model.


# How did we get the Decision Tree

# When building the Decision Tree, we don’t just randomly choose which feature to split on first. 
# We want to start by choosing the feature with the most predictive power. Let’s look at our same Decision Tree again.

# Intuitively for our Titanic dataset, since women were often given priority on lifeboats, we expect the Sex to be a very important feature. 
# So using this feature first makes sense. On each side of the Decision Tree, we will independently determine which feature to split on next. 
# In our example above, the second split for women is on Pclass. The second split for men is on Age. 
# We also note for some cases we do three splits and for some just two.
# For any given dataset, there’s a lot of different possible Decision Trees that could be created depending on the order you use the features. 
# In the upcoming lessons, we’ll see how to mathematically choose the best Decision Tree.
# What makes a Good Split


# In order to determine which feature we should split on first, we need to score every possible split so we can choose the split with 
# the highest score. Our goal would be to perfectly split the data. If, for instance, all women survived the crash and all men didn’t survive, 
# splitting on Sex would be a perfect split. This is rarely going to happen with a real dataset, but we want to get as close to this as possible.

# The mathematical term we’ll be measuring is called information gain. This will be a value from 0 to 1 where 0 is the information gain of a 
# useless split and 1 is the information gain of a perfect split. In the next couple parts we will define gini impurity and entropy which 
# we will use to define information gain. First we will discuss the intuition of what makes a good split.

# Let’s consider a couple possible splits for the Titanic dataset. We’ll see how it splits the data and why one is better than the other.

# First, let’s trying splitting on Age. Since Age is a numerical feature, we need to pick a threshold to split on. Let’s say we split on 
# Age<=30 and Age>30. Let’s see how many passengers we have on each side, and how many of them survived and how many didn’t.
# On both sides, we have about 40% of the passengers surviving. Thus we haven’t really gained anything from splitting the data this way.

# Now let’s try splitting on Sex.
# We can see on the female side that the vast majority survived. On the male side, the vast majority didn’t survive. This is a good split.

# What we’re going for is homogeneity (or purity) on each side. Ideally we would send all the passengers who survived to one side and 
# those who didn’t survive to the other side. We’ll look at two different mathematical measurements of purity. We’ll use the purity values 
# to calculate the information gain.
# A good choice of a feature to split on results in each side of the split being pure. A set is pure if all the datapoints belong to the 
# same class (either survived or didn’t survive).


# Gini Impurity

# Gini impurity is a measure of how pure a set is. We’ll later see how we can use the gini impurity to calculate the information gain.

# We calculate the gini impurity on a subset of our data based on how many datapoints in the set are passengers that survived and 
# how many are passengers who didn’t survive. It will be a value between 0 and 0.5 where 0.5 is completely impure 
# (50% survived and 50% didn’t survive) and 0 is completely pure (100% in the same class).

# The formula for gini is as follows. p is the percent of passengers who survived. Thus (1-p) is the percent of passengers who didn’t survive.

# GINI = 2 * p * (1-p)

# We can see that the maximum value is 0.5 when exactly 50% of the passengers in the set survived. 
# If all the passengers survived or didn’t survive (percent is 0 or 1), then the value is 0.

# Let’s calculate the gini impurity for our examples from the previous part. First we had a split on Age<=30 and Age>30. 
# Let’s calculate the gini impurities of the two sets that we create.

# On the left, for the passengers with Age<=30, let’s first calculate the percent of passengers who survived:

# Percent of passengers who survived = 197/(197+328) = 0.3752
# Percent of passengers who didn’t survive = 1 - 0.375 = 0.6248

# Now let’s use that to calculate the gini impurity:
# 2 * 0.3752 * 0.6248 = 0.4689
# We can see that this value is close to 0.5, the maximum value for gini impurity. This means that the set is impure.

# Now let’s calculate the gini impurity for the right side, passengers with Age>30.
# 2 * 145/(145+217) * 217/(145+217) = 0.4802
# This value is also close to 0.5, so again we have an impure set.

# Now let’s look at the gini values for the other split we tried, splitting on Sex.

# On the left side, for female passengers, we calculate the following value for the gini impurity.
# 2 * 233/(233+81) * 81/(233+81) = 0.3828
# On the right side, for male passengers, we get the following value.
# 2 * 109/(109+464) * 464/(109+464) = 0.3081
# Both of these values are smaller than the gini values for splitting on Age, so we determine that splitting on the Sex feature is a better choice.
# Right now we have two values for each potential split. The information gain will be a way of combining them into a single value.


# Entropy

# Entropy is another measure of purity. It will be a value between 0 and 1 where 1 is completely impure (50% survived and 50% didn’t survive) 
# and 0 is completely pure (100% the same class).
# The formula for entropy comes from physics. p again is the percent of passengers who survived.

# entropy = -[p log2 p + (1 - p) log2 (1 - p)]

# You can see it has a similar shape to the gini function. Like the gini impurity, the maximum value is when 50% of the passengers 
# in our set survived, and the minimum value is when either all or none of the passengers survived. The shape of the graphs are a little different. 
# You can see that the entropy graph is a little fatter.

# Now let’s calculate the entropy values for the same two potential splits.

# On the left (Age<=30):
# p = 197/(197+328) = 0.3752
# Entropy = -(0.375 * log(0.375) + (1-0.375) * log(1-0.375)) = 0.9546

# And on the right (Age>30):
# p = 145/(145+217) = 0.4006
# Entropy =  -(0.401 * log(0.401) + (1-0.401) * log(1-0.401)) =  0.9713

# These values are both close to 1, which means the sets are impure.
# Now let’s do the same calculate for the split on the Sex feature.

# On the left (female):
# p = 233/(233+81) = 0.7420
# Entropy = -(p * log(p) + (1-p) * log(1-p)) = 0.8237

# And on the right (male):
# p = 109/(109+464) = 0.1902
# Entropy =  -(p * log(p) + (1-p) * log(1-p)) = 0.7019

# You can see that these entropy values are smaller than the entropy values above, so this is a better split.
# It’s not obvious whether gini or entropy is a better choice. It often won’t make a difference, but you can always cross validate to compare a 
# Decision Tree with entropy and a Decision Tree with gini to see which performs better.


# Information Gain

# Now that we have a way of calculating a numerical value for impurity, we can define information gain.

# Information Gain = H(S) - |A|/|S| H(A) - |B|/|S| H(B)

# H is our impurity measure (either Gini impurity or entropy). S is the original dataset and A and B are the two sets we’re splitting 
# the dataset S into. In the first example above, A is passengers with Age<=30 and B is passengers with Age>30. In the second example, 
# A is female passengers and B is male passengers. |A| means the size of A.

# Let’s calculate this value for our two examples. Let’s use Gini impurity as our impurity measure.

# We’ve already calculated most of the Gini impurity values, though we need to calculate the Gini impurity of the whole set. 
# There are 342 passengers who survived and 545 passengers who didn’t survive, out of a total of 887 passengers, 
# so the gini impurity is as follows:
# Gini = 2 * 342/887 * 545/887 = 0.4738
# Again, here’s the first potential split.

# Note that we have 197+328=525 passengers on the left (Age<=30) and 145+217=362 passengers on the right (Age>30). 
# Thus, pulling in the gini impurity values that we calculated before, we get the following information gain:
# Information gain = 0.4738 - 525/887 * 0.4689 - 362/887 * 0.4802 = 0.0003
# This value is very small meaning we gain very little from this split.

# Now let’s calculate the information gain for splitting on Sex.
# We have 233+81=314 passengers on the left (female) and 109+464=573 passengers on the right (male). Here is the information gain:
# Information gain = 0.4738 - 314/887 * 0.3828 - 573/887 * 0.3081 = 0.1393

# Thus we can see that the information gain is much better for this split. Therefore, splitting on Sex 
# is a much better choice when building our decision tree than splitting on Age with threshold 30.
# The work we did was just to compare two possible splits. We’ll need to do the same calculations for every possible split in order 
# to find the best one. Luckily we don’t have to do the computations by hand!


# Building the Decision Tree

# We’ve built up the foundations we need for building the Decision Tree. Here’s the process we go through:

# To determine how to do the first split, we go over every possible split and calculate the information gain if we used that split. 
# For numerical features like Age, PClass and Fare, we try every possible threshold. Splitting on the Age threshold of 50 means that datapoints 
# with Age<=50 are one group and those with Age>50 are the other. Thus since there are 89 different ages in our dataset, we have 88 
# different splits to try for the age feature!

# We need to try all of these potential splits:
# 1. Sex (male | female)
# 2. Pclass (1 or 2 | 3)
# 3. Pclass (1 | 2 or 3)
# 4. Age (0 | >0)
# 5. Age (<=1 | >1)
# 6. Age (<=2 | >2)
# 7. etc….

# There is 1 potential split for Sex, 2 potential splits for Pclass, and 88 potential splits for Age. There are 248 different values for Fare, 
# so there are 247 potential splits for this feature. If we’re only considering these four features, we have 338 potential splits to consider.

# For each of these splits we calculate the information gain and we choose the split with the highest value.

# Now, we do the same thing for the next level. Say we did the first split on Sex. Now for all the female passengers, we try all of the possible 
# splits for each of the features and choose the one with the highest information gain. We can split on the same feature twice if that feature 
# has multiple possible thresholds. Sex can only be split on once, but the Fare and Age features can be split on many times.

# Independently, we do a similar calculation for the male passengers and choose the split with the highest information gain. Thus we may have a 
# different second split for male passengers and female passengers.

# We continue doing this process until we have no more features to split on.
# This is a lot of things to try, but we just need to throw computation power at it. It does make Decision Trees a little slow to build, but once 
# the tree is built, it is very fast to make a prediction.



# Decision Tree Diagram

# Let’s look at an example Decision Tree for the Titanic dataset. Within each internal node, we have the feature and threshold to split on, 
# the number of samples and the distribution of the sames (# didn’t survived vs survived).

# To interpret this, let’s start by looking at the root node. It says:

# male <= 0.5
# samples = 887
# value = [545, 342]

# This means that the first split will be on the male column. If the value is <=0.5 (meaning the passenger is female) we go to the left 
# child and if the value is >0.5 (meaning the passenger is male) we go to the right child.

# There are 887 datapoints to start and 545 are negative cases (didn’t survive) and 342 are positive (did survive).

# If you look at the two children of the root node, we can see how many datapoints were sent each way based on splitting on Sex. 
# There are 314 female passengers in our dataset and 573 male passengers.

# You can see that the second split for female passengers is different from the second split for male passengers.
# This diagram was created with graphviz, which we’ll learn how to use in a later lesson.


# How to Make a Prediction

# Let’s look at the same decision tree diagram again.

# Let’s say we’d like to use this Decision Tree to make a prediction for a passenger with these values:

# Sex: female
# Pclass: 3
# Fare: 25
# Age: 30

# We ask the question at each node and go to the left child if the answer is yes and to the right if the answer is no.

# We start at the root node.

# Is the value for the male feature <= 0.5? (This question could also be asked as "Is the passenger female?")
# Since the answer is yes, we go to the left child.

# Is the Pclass <= 0.5?
# Since the answer is no, we go to the right child.

# Is the Fare <= 23.35?
# Since the answer is no, we go to the right child.

# Now we’re at a leaf node. Here’s the path we took highlighted.

# The leaf node that we end at has the following text:
# 27
# [24, 3]
# This means there are 27 datapoints in our dataset that also land at this leaf node. 24 of them didn’t survive 
# and 3 of them survived. This means our prediction is that the passenger didn’t survive.
# Because there are no rules as to how the tree is developed, the decision tree asks completely different questions 
# of a female passenger than a male passenger.



# DecisionTreeClassifier Class

# Just like with Logistic Regression, scikit-learn has a Decision Tree class. The code for building 
# a Decision Tree model is very similar to building a Logistic Regression model. Scikit-learn did this intentionally so that 
# it is easy to build and compare different models for the same dataset.

# Here’s the import statement.
from sklearn.tree import DecisionTreeClassifier

# Now we can apply the same methods that we used with the LogisticRegression class: fit (to train the model), score 
# (to calculate the accuracy score) and predict (to make predictions).

# We first create a DecisionTreeClassifier object.
model = DecisionTreeClassifier()

# We do a train/test split using a random_state so that every time we run the code we will get the same split.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)

# Then we use the fit method to train the model.
model.fit(X_train, y_train)

# We can use the predict method to see what the model predicts. Here we can see the prediction for a male passenger in 
# Pclass 3, who is 22 years old, has 1 sibling/spouse on board, has 0 parents/children on board, and paid a fare of 7.25.

print(model.predict([[3, True, 22, 1, 0, 7.25]]))

# We see that the model predicts that the passenger did not survive. This is the same prediction that the Logistic Regression model gave.
# Note that we have the same methods for a DecisionTreeClassifier as we did for a LogisticRegression object.

# Scoring a Decision Tree Model

# We can use the score and predict methods to get the accuracy, precision and recall scores.

print("Accuracy: ", model.score(X_test, y_test))
y_pred = model.predict(X_test)
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))

# We can use k-fold cross validation to get an accurate measure of the metrics and compare the values with a Logistic Regression model. 
# We use a random_state when creating the KFold object so that we will get the same results every time.
# You can see that the accuracy and precision of the Logistic Regression model is higher, and the recalls of the two models are about the same.
# The Logistic Regression model performs better, though we may still want to use a Decision Tree for its interpretability.


# Gini vs Entropy

# The default impurity criterion in scikit-learn’s Decision Tree algorithm is the Gini Impurity. However, they’ve also implemented entropy 
# and you can choose which one you’d like to use when you create the DecisionTreeClassifier object.

# If you go to the docs, you can see that one of the parameters is criterion.

# sklearn.tree.DecisionTreeClassifier
# class sklearn. tree. DecisionT reeClassifier(criterion='gini', splitter="best
# max depth=None, min_samples_split=2, min_samples leaf=1,
# min weight fraction_leaf=0.0, max features=None, random_state=None,
# maxleatnodes=None, min_impurity decrease=0.0, min_impurity_split=None,
# class weight=None, presort='deprecated', ccp_alpha=0.0)

# Parameters:
# criterion: str, optional (default="gini")
# The function to measure the quality of a split. Supported
# criteria are "gini" for the Gini impurity and "entropy" for the
# information gain.

# The docs are located here.
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

# To build a Decision Tree that uses entropy, we’ll need to set the criterion parameter to entropy. 
# Here’s the code for building a Decision Tree that uses entropy instead of the Gini Impurity.
# dt = DecisionTreeClassifer(criterion='entropy')

# Now we can compare a Decision Tree using gini with a Decision Tree using entropy. We first create a k-fold split since when we’re 
# comparing two models we want them to use the same train/test splits to be fair. Then we do a k-fold cross validation with each of the two 
# possible models. We calculate accuracy, precision and recall for each of the two options.

kf = KFold(n_splits=5, shuffle=True)
for criterion in ['gini', 'entropy']:
    print("Decision Tree - {}".format(criterion))
    accuracy = []
    precision = []
    recall = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dt = DecisionTreeClassifier(criterion=criterion)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
    print("accuracy: ", np.mean(accuracy))
    print("precision: ", np.mean(precision))
    print("recall: ", np.mean(recall))

# We see very little difference in the performance of Gini vs Entropy. This is expected as they aren’t really very different functions. 
# It’s rare to find a dataset where the choice would make a difference.



# Visualizing Decision Trees

# If you want to create a png image of your graph, like the ones shown in this module, you can use scikit-learn's export_graphviz function.

# First we import it.
from sklearn.tree import export_graphviz

# dot_file = export_graphviz(dt, feature_names=feature_names)

# Then we use the export_graphviz function. Here dt is a Decision Tree object and feature_names is a list of the feature names. 
# Graph objects are stored as .dot files which can be the GraphViz program. Our goal is to save a png image file. We will be able to 
# convert the dot file to a png file, so we first save the dot file to a variable., so we save the dot file created by the export_graphviz 
# function so that we can convert it to a png.

# We can then use the graphviz module to convert it to a png image format.
import graphviz
# graph = graphviz.Source(dot_file)

# Finally, we can use the render method to create the file. We tell it the filename and file format. By default, it will create extra 
# files that we are not interested in, so we add cleanup to tell it to get rid of them.
# graph.render(filename='tree', format='png', cleanup=True)

# Now you should have a file called tree.png on your computer. Here's the code for visualizing the tree for the Titanic dataset with 
# just the Sex and Pclass features.
from sklearn.tree import export_graphviz
import graphviz
from IPython.display import Image

feature_names = ['Pclass', 'male']
X = df[feature_names].values
y = df['Survived'].values

dt = DecisionTreeClassifier()
dt.fit(X, y)

dot_file = export_graphviz(dt, feature_names=feature_names)
graph = graphviz.Source(dot_file)
graph.render(filename='tree', format='png', cleanup=True)

# If you're going to run this on your computer, make sure to install graphviz first. You can do this with "pip install graphviz".

from sklearn.tree import export_graphviz
import graphviz
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from IPython.display import Image

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'

feature_names = ['Pclass', 'male']
X = df[feature_names].values
y = df['Survived'].values

dt = DecisionTreeClassifier()
dt.fit(X, y)

dot_file = export_graphviz(dt, feature_names=feature_names)
graph = graphviz.Source(dot_file)
graph.render(filename='tree', format='png', cleanup=True)


# Tendency to Overfit

# Recall that overfitting is when we do a good job of building a model for the training set, but it doesn’t perform well on the test set. 
# Decision Trees are incredibly prone to overfitting. Since they can keep having additional nodes in the tree that split on features, 
# the model can really dig deep into the specifics of the training set. Depending on the data, this might result in a model that doesn’t 
# capture the true nature of the data and doesn’t generalize.

# Maybe we just have a single datapoint that goes to a leaf node. It might not make sense to have that additional split.

# Let’s look at a diagram of a Decision Tree for the Titanic dataset. This is the resulting tree when we build a Decision Tree with scikit-learn 
# on the entire dataset. We’re just looking at a portion of the Decision Tree since it’s so large. We’ve highlighted a particular path of interest.

# If you follow the highlighted path, you’ll see that we split on Sex, Pclass, and then split on Age 9 times in a row with different thresholds. 
# This results in a graph that’s very nitpicky about age. A female passenger in Pclass 3 of age 31 goes to a different leaf node than a similar 
# passenger of age 30.5 or 30 or 29. The model predicts that a female passenger age 35 survives, age 32 doesn’t survive, age 31 survives, 
# and age 30 doesn’t survive. This is probably too fine-grained and is giving single datapoints from our dataset too much power. You can see 
# that the leaf nodes all have few datapoints and often only one.
# If you let a Decision Tree keep building, it may create a tree that’s overfit and doesn’t capture the essence of the data.


# Pruning

# In order to solve these issues, we do what’s called pruning the tree. This means we make the tree smaller with the goal of reducing overfitting.

# There are two types of pruning: pre-pruning & post-pruning. In pre-pruning, we have rules of when to stop building the tree, so we stop building 
# before the tree is too big. In post-pruning we build the whole tree and then we review the tree and decide which leaves to remove to make the tree smaller.
# The term pruning comes from the same term in farming. Farmers cut off branches of trees and we are doing the same to our decision tree.

# Pre-pruning

# We’re going to focus on pre-pruning techniques since they are easier to implement. We have a few options for how to limit the tree growth. 
# Here are some commonly used pre-pruning techniques:
# • Max depth: Only grow the tree up to a certain depth, or height of the tree. If the max depth is 3, there will be at most 3 splits for each datapoint.
# • Leaf size: Don’t split a node if the number of samples at that node is under a threshold
# • Number of leaf nodes: Limit the total number of leaf nodes allowed in the tree

# Pruning is a balance. For example, if you set the max depth too small, you won’t have much of a tree and you won’t have any predictive power. 
# This is called underfitting. Similarly if the leaf size is too large, or the number of leaf nodes too small, you’ll have an underfit model.
# There’s no hard science as to which pre-pruning method will yield better results. In practice, we try a few different values for each parameter 
# and cross validate to compare their performance.


# Pre-pruning Parameters

# Scikit-learn has implemented quite a few techniques for pre-pruning. In particular, we will look at three of the parameters: max_depth, min_samples_leaf, 
# and max_leaf_nodes. Look at the docs for Decision Trees to find full explanations of these three parameters.

# Prepruning Technique 1: Limiting the depth

# We use the max_depth parameter to limit the number of steps the tree can have between the root node and the leaf nodes.

# Prepruning Technique 2: Avoiding leaves with few datapoints

# We use the min_samples_leaf parameter to tell the model to stop building the tree early if the number of datapoints in a leaf will be below a threshold.

# Prepruning Technique 3: Limiting the number of leaf nodes

# We use max_leaf_nodes to set a limit on the number of leaf nodes in the tree.

# Here’s the code for creating a Decision Tree with the following properties:
# • max depth of 3
# • minimum samples per leaf of 2
# • maximum number of leaf nodes of 10
dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, max_leaf_nodes=10)

# You can now train the model and test it as we’ve done before.

# You can use as many or as few of the parameters as you’d like.
# To determine the best values for the pre-pruning parameters, we’ll use cross validation to compare several potential options.


# Grid Search

# We’re not going to be able to intuit best values for the pre-pruning parameters. In order to decide on which to use, we use cross validation 
# and compare metrics. We could loop through our different options like we did in the Lesson on Decision Trees in Scikit-learn, but scikit-learn 
# has a grid search class built in that will do this for us.

# The class is called GridSearchCV. We start by importing it.
from sklearn.model_selection import GridSearchCV

# GridSearchCV has four parameters that we’ll use:
# 1. The model (in this case a DecisionTreeClassifier)
# 2. Param grid: a dictionary of the parameters names and all the possible values
# 3. What metric to use (default is accuracy)
# 4. How many folds for k-fold cross validation

# Let’s create the param grid variable. We’ll give a list of all the possible values for each parameter that we want to try.
param_grid = {
    'max_depth': [5, 15, 25],
    'min_samples_leaf': [1, 3],
    'max_leaf_nodes': [10, 20, 35, 50]}

# Now we create the grid search object. We’ll use the above parameter grid, set the scoring metric to the F1 score, and do a 5-fold cross validation.
dt = DecisionTreeClassifier()
gs = GridSearchCV(dt, param_grid, scoring='f1', cv=5)

# Now we can fit the grid search object. This can take a little time to run as it’s trying every possible combination of the parameters.
gs.fit(X, y)

# Since we have 3 possible values for max_depth, 2 for min_samples_leaf and 4 for max_leaf_nodes, we have 3 * 2 * 4 = 24 different combinations to try:

# max_depth: 5, min_samples_leaf: 1, max_leaf_nodes: 10
# max_depth: 15, min_samples_leaf: 1, max_leaf_nodes: 10
# max_depth: 25, min_samples_leaf: 1, max_leaf_nodes: 10
# max_depth: 5, min_samples_leaf: 3, max_leaf_nodes: 10
# ...

# We use the best_params_ attribute to see which model won.

print("Best params: ", gs.best_params_)

# Thus we see that the best model has a maximum depth of 15, maximum number of leaf nodes as 35 and minimum samples per leaf of 1.

# The best_score_ attribute tells us the score of the winning model.

print("best score: ", gs.best_score_)

# There are often a few models that have very similar performance. If you run this multiple times you might get slightly different results depending 
# on the randomness of how the points are distributed among the folds. Generally if we have multiple models with comparable performance, 
# we’d choose the simpler model.

# Computation

# When talking about how much computation is required for a machine learning algorithm, we separate it into two questions: how much computation 
# is required to build the model and how much is required to predict.

# A decision tree is very computationally expensive to build. This is because at every node we are trying every single feature and threshold as a 
# possible split. We have to calculate the information gain of each of these possible splits each time. This is computationally very expensive.

# Predicting with a decision tree on the other hand, is computational very inexpensive. You just need to ask a series of yes/no questions about the 
# datapoint to get to the prediction.
# Generally we care much more about the computation time for prediction than training. Predictions often need to happen in real time while a user 
# is waiting for a result.

# Performance

# Decision Trees can perform decently well depending on the data, though as we have discussed, they are prone to overfitting. 
# Since a leaf node can have just one datapoint that lands there, it gives too much power to individual datapoints.

# To remedy the overfitting issues, decision trees generally require some tuning to get the best possible model. 
# Pruning techniques are used to limit the size of the tree and they help mitigate overfitting.
# Decision Trees often take work to perform on par with how other models perform with no tuning.

# Interpretability


# The biggest reason that people like choosing decision trees is because they are easily interpretable. Depending on what you’re building a model for, 
# you might need to give a reason why you made a certain prediction. A non-technical person can interpret a Decision Tree so it’s easy to give an 
# explanation of a prediction.

# This particularly comes into play in legal situations. Say you are a bank and have a model to predict whether a person should be given a loan or not. 
# It is important to be able to explain why the model made the decision, otherwise you could hide discriminatory practices within the model.
# Interpretability is the biggest advantage of Decision Trees. It will depend on the situation whether this is important for your problem.


# Improving on Decision Trees


# As we learned in the previous module, the main drawback of decision trees are that they have a tendency to overfit. We saw that we could improve 
# their performance with pruning, but in this module we’ll see a way of using decision trees to make a better model.

# Decision Trees are very susceptible to random idiosyncrasies in the training dataset. We say that Decision Trees have high variance since if you 
# randomly change the training dataset, you may end up with a very different looking tree.

# One of the advantages of decision trees over a model like logistic regression is that they make no assumptions about how the data is structured. 
# In logistic regression, we assume that we can draw a line to split the data. Sometimes our data just isn’t structured like that. 
# A decision tree has the potential to get at the essence of the data no matter how it is structured.

# We will be learning about random forests in this module, which as you may guess from the name, is a model built with multiple trees. 
# The goal of random forests is to take the advantages of decision trees while mitigating the variance issues.

# A random forest is an example of an ensemble because it uses multiple machine learning models to create a single model.


# Bagging Decision Trees

# Bootstrap Aggregation (or Bagging) is a technique for reducing the variance in an individual model by creating an ensemble from multiple models built on bootstrapped samples.

# To bag decision trees, we create multiple (say 10) bootstrapped resamples of our training dataset. So if we have 100 datapoints in our training set, each of the resamples 
# will have 100 datapoints randomly chosen from our training set. Recall that we randomly select with replacement, meaning that some datapoints will appear multiple times and some not at all.

# We create a decision tree with each of these 10 resamples.

# To make a prediction, we make a prediction with each of the 10 decision trees and then each decision tree gets a vote. The prediction with the most votes is the final prediction.

# When we bootstrap the training set, we're trying to wash out the variance of the decision tree. The average of several trees that have different training sets will create a 
# model that more accurately gets at the essence of the data.
# Bagging Decision Trees is a way of reducing the variance in the model.


# Decorrelate the Trees

# With bagged decision trees, the trees may still be too similar to have fully created the ideal model. They are built on different resamples, but they all have access to the same features. 
# Thus we will add some restrictions to the model when building each decision tree so the trees have more variation. We call this decorrelating the trees.

# If you recall, when building a decision tree, at every node, we compare all the split thresholds for each feature to find the single best feature & split threshold. In a decision tree 
# for a random forest, at each node, we randomly select a subset of the features to consider. This will result in us choosing a good, but not the best, feature to split on at each step. 
# It’s important to note that the random selection of features happens at each node. So maybe at the first node we consider the Sex and Fare features and then at the second node, 
# the Fare and Age features.

# A standard choice for the number of features to consider at each split is the square root of the number of features. So if we have 9 features, we will consider 3 of them at each node 
# (randomly chosen).

# If we bag these decision trees, we get a random forest.
# Each decision tree within a random forest is probably worse than a standard decision tree. But when we average them we get a very strong model!


# Review of Breast Cancer Dataset

# We’ll be using the breast cancer dataset in this lesson. Recall that the dataset has measures of different attributes of a lump in breast tissue and a label of whether or not the tumor is cancerous.

# Here’s the code for pulling the data from sklearn:

import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data['data'], columns = cancer_data['feature_names'])
df['target'] = cancer_data['target']

X = df[cancer_data.feature_names].values
y = df['target'].values
print ('data dimensions', X.shape)


# Random Forest with Sklearn


# The syntax for building and using a Random Forest model is the same as it was for Logistic Regression and Decision Trees. 
# The builders of scikit-learn intentionally made it so that it would be easy to switch between and compare different models.

# Here’s the import statement for importing the Random Forest model for classification.
from sklearn.ensemble import RandomForestClassifier

# We’ll first split the dataset into a training set and test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)

# We have added the random state parameter here so that it will do the same split every time we run the code. Without the random state, 
# we’d expect different datapoints in the training and testing sets each time we run the code which can make it harder to test the code.

# Then we create the RandomForestClassifier object and use the fit method to build the model on the training set.
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Now we can use the model to make a prediction. For example, let’s take the first row of the test set and see what the prediction is. 
# Recall that the predict method takes an array of points, so even when we have just one point, we have to put it in a list.

first_row = X_test[0]
print("Prediction: ", rf.predict([first_row]))
print("True value: ", y_test[0])

# These results mean that the model predicted that the lump was cancerous and that was correct.

# We can use the score method to calculate the accuracy over the whole test set.

print("random forest accuracy: ", rf.score(X_test, y_test))

# Thus the accuracy is 96.5%. We can see how this compares to the Decision Tree model.

dt = DecisionTreeClassifier
dt.fit(X_train, y_train)
print("Decision tree accuracy: ", dt.score(X_test, y_test))

# So the accuracy for the decision tree model is 90.2%, much worse than that for the random forest.
# Note how similar the scikit-learn code is to the code for Logistic Regression and Decision Trees. 
# This makes it very easy to try and compare different models.

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data['data'], columns = cancer_data['feature_names'])
df['target'] = cancer_data['target']

X = df[cancer_data.feature_names].values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

first_row = X_test[0]
print("Prediction: ", rf.predict([first_row]))
print("True value: ", y_test[0])

print("Random forest accuracy: ", rf.score(X_test, y_test))

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print("Decision tree accuracy: ", dt.score(X_test, y_test))



# Random Forest Parameters


# When you look at the scikit-learn docs for the RandomForestClassifier, you will see quite a few parameters that you can control. 
# We will be looking at a few of them, but you can learn about all of them in the docs.

# Since a random forest is made up of decision trees, we have all the same tuning parameters for prepruning as we did for decision trees: 
# max_depth, min_samples_leaf, and max_leaf_nodes. With random forests, it is generally not important to tune these as overfitting is 
# generally not an issue.

# We will look at two new tuning parameters: n_estimators (the number of trees) and max_features (the number of features to consider at each split).

# The default for the max features is the square root of p, where p is the number of features (or predictors). The default is generally 
# a good choice for max features and we usually will not need to change it, but you can set it to a fixed number with the following code.
rf = RandomForestClassifier(max_features=5)

# The default number of estimators (decision trees) is 10. This often works well but may in some cases be too small. 
# You can set it to another number as follows. We will see in the next parts how to choose the best value.
rf = RandomForestClassifier(n_estimators=15)

# One of the big advantages of Random Forests is that they rarely require much tuning. The default values will work well on most datasets.


# Grid Search

# If you recall from the Decision Tree module, scikit-learn has built in a Grid Search class to help us find the optimal choice of parameters.

# Let’s use Grid Search to compare the performance of a random forest with different numbers of trees.

# Recall that we need to define the parameter grid of the parameters we want to vary and give a list of the values to try.
param_grid = {
    'n_estimators': [10, 25, 50, 75, 100],
}

# Now we can create a Random Forest Classifier and a Grid Search. Recall that the Grid Search will do k-fold cross validation for us. 
# We set cv=5 for 5-fold cross validation.
rf = RandomForestClassifier()
gs = GridSearchCV(rf, param_grid, cv=5)

# Now we use the fit method to run the grid search. The best parameters will be stored in the best_params_ attribute.
gs.fit(X, y)
print("best params:", gs.best_params_)
# best params: {'n_estimators': 50}

# These are the parameters which yield the highest accuracy as that is the default metric. Note that you may get slightly different results 
# each time you run this as the random split in the 5 folds may affect which has the best accuracy score.

# Accuracy will work okay for us in this case as the classes in the breast cancer dataset are reasonably balanced. If the classes are imbalanced, 
# we would want to use an alternative metric, like the f1-score. We can change the metric by scoring parameter to "f1" as follows. 
# To avoid outputting a different best parameter each time, one can set the random_state in the classifier.

rf = RandomForestClassifier(random_state=123)
gs = GridSearchCV(rf, param_grid, scoring='f1', cv=5)
gs.fit(X,y)
print("Best params: ", gs.best_params_)

# You can add additional parameters, e.g. max_features, and parameter values to the param_grid dictionary to compare more decision trees.

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data['data'], columns = cancer_data['feature_names'])
df['target'] = cancer_data['target']

X = df[cancer_data.feature_names].values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)
param_grid = {
    'n_estimators': [10, 25, 50, 75, 100],
}
rf = RandomForestClassifier(random_state=123)
gs = GridSearchCV(rf, param_grid, scoring='f1', cv=5)
gs.fit(X,y)
print("Best params: ", gs.best_params_)


# Elbow Graph

# With a parameter like the number of trees in a random forest, increasing the number of trees will never hurt performance. 
# Increasing the number trees will increase performance until a point where it levels out. The more trees, however, the more complicated the algorithm. 
# A more complicated algorithm is more resource intensive to use. Generally it is worth adding complexity to the model if it improves performance 
# but we do not want to unnecessarily add complexity.

# We can use what is called an Elbow Graph to find the sweet spot. Elbow Graph is a model that optimizes performance without adding unnecessary complexity.

# To find the optimal value, let’s do a Grid Search trying all the values from 1 to 100 for n_estimators.

n_estimators = list(range(1, 101))
param_grid = {
    'n_estimators': n_estimators,
}
rf = RandomForestClassifier()
gs = GridSearchCV(rf, param_grid, cv=5)
gs.fit(X, y)

# Instead of just looking at the best params like we did before, we are going to use the entire result from the grid search. 
# The values are located in the cv_results_ attribute. This is a dictionary with a lot of data, however, we will only need one of the keys: 
# mean_test_score. Let’s pull out these values and store them as a variable.
scores = gs.cv_results_['mean_test_score']
# [0.91564148, 0.90685413, ...]

# Now let’s use matplotlib to graph the results.
import matplotlib.pyplot as plt

scores = gs.cv_results_['mean_test_score']
plt.plot(n_estimators, scores)
plt.xlabel("n_estimators")
plt.ylabel("accuracy")
plt.xlim(0, 100)
plt.ylim(0.9, 1)
plt.show()

# If we look at this graph, we see that around 10 trees the graph levels out. The best model occurred at n_estimators=33 and n_estimators=64, 
# but given how volatile it is, that was probably due to random chance. We should choose about 10 to be our number of estimators, 
# because we want the minimum number of estimators that still yield maximum performance.

# Now we can build our random forest model with the optimal number of trees.
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X, y) 

# You’ll see elbow graphs pop up in lots of different situations when we are adding complexity to a model and 
# want to determine the minimal amount of complexity that will yield optimal performance.


# Feature Importances

# There are 30 features in the cancer dataset. Does every feature contribute equally to building a model? 
# If not, which subset of features should we use? This is a matter of feature selection.

# Random forests provide a straightforward method for feature selection: mean decrease impurity. 
# Recall that a random forest consists of many decision trees, and that for each tree, the node is chosen to split the dataset 
# based on maximum decrease in impurity, typically either Gini impurity or entropy in classification. Thus for a tree, 
# it can be computed how much impurity each feature decreases in a tree. And then for a forest, the impurity decrease 
# from each feature can be averaged. Consider this measure a metric of importance of each feature, we then can rank and select the 
# features according to feature importance.

# Scikit-learn provides a feature_importances_ variable with the model, which shows the relative importance of each feature. 
# The scores are scaled down so that the sum of all scores is 1.
# Let's find the feature importances in a random forest with n_estimator = 10 using the training dataset, and display them in the descending order.

rf = RandomForestClassifier(n_estimators=10, random_state=111)
rf.fit(X_train, y_train)

ft_imp = pd.Series(rf.feature_importances_, index=cancer_data.feature_names).sort_values(ascending=False)
ft_imp.head(10)

# From the output, we can see that among all features, worst radius is most important (0.31), followed by mean concave points and worst concave points.
# In regression, we calculate the feature importance using variance instead.


# New Model on Selected Features

# Why should we perform feature selection? Top reasons are: it enables us to train a model faster; it reduces the complexity of a model 
# thus makes it easier to interpret. And if the right subset is chosen, it can improve the accuracy of a model. Choosing the right subset 
# often relies on domain knowledge, some art, and a bit of luck.

# In our dataset, we happen to notice that features with "worst" seem to have higher importances. As a result we are going to build a new 
# model with the selected features and see if it improves accuracy. Recall the model from the last part.

rf = RandomForestClassifier(n_estimators=10, random_state=111)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

# We first find the features whose names include the word "worst":

worst_cols = [col for col in df.columns if 'worst' in col]
print(worst_cols)

# There are ten such features. Now we create another dataframe with the selected features, followed by a train test split with the same random state.

X_worst = df[worst_cols]
X_train, X_test, y_train, y_test = train_test_split(X_worst, y, random_state=101)

# Fit the model and output the accuracy.

rf.fit(X_train, y_train)
rf.score(X_test, y_test)

# Here we are able to improve the accuracy using a subset of features, a third of the total features to be exact. 
# This is because we removed some noise and highly correlated features, resulting in an increased accuracy. 
# The advantage of building a better model using less features will be more pronounced when the sample size is large.
# There is no best feature selection method, at least not universally. Instead, we must discover what works best for 
# the specific problem and leverage the domain expertise to build a good model.
# Scikit-learn provides an easy way to discover the feature importances.


# Performance

# Probably the biggest advantage of Random Forests is that they generally perform well without any tuning. 
# They will also perform decently well on almost every dataset.

# A linear model, for example, cannot perform well on a dataset that cannot be split with a line. It is not possible to 
# split the following dataset with a line without manipulating the features. However, a random forest will perform just fine on this dataset.

# We can see this by looking at the code to generate the fake dataset above and comparing a Logistic Regression model with a Random Forest model. 
# The function make_circles makes a classification dataset with concentric circles. We use kfold cross validation to compare the accuracy scores 
# and see that the Logistic Regression model performs worse than random guessing but the Random Forest model performs quite well.

from sklearn.datasets import make_circles
from sklearn.model_selection import KFold
from sklearn. linear_model import LogisticRegression
from sklearn. ensemble import RandomForestClassifier
import numpy as np

X, y = make_circles(noise=0.2, factor=0.5, random_state=1)

kf = KFold(n_splits=5, shuffle=True, random_state=1)
lr_scores = []
rf_scores = []

for train_index, test_index in kf.split(X) :
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index] , y[test_index]

    lr = LogisticRegression( solver= 'lbfgs')
    lr.fit(X_train, y_train)
    lr_scores.append(lr.score(X_test, y_test))

    rf = RandomForestClassifier(n_estimators = 100)
    rf.fit(X_train, y_train)
    rf_scores.append(rf.score(X_test, y_test))

print("LR accuracy:", np.mean (lr_scores))
print("RF accuracy:", np.mean(rf_scores))

# When looking to get a benchmark for a new classification problem, it is common practice to start by building a Logistic Regression 
# model and a Random Forest model as these two models both have potential to perform well without any tuning. This will give you values 
# for your metrics to try to beat. Oftentimes it is almost impossible to do better than these benchmarks.


# Interpretability

# Random Forests, despite being made up of Decision Trees, are not easy to interpret. A random forest has several decision trees, 
# each of which is not a very good model, but when averaged, create an excellent model. Thus Random Forests are not a good choice 
# when looking for interpretability.
# In most cases, interpretability is not important.


# Computation

# Random Forests can be a little slow to build, especially if you have a lot of trees in the random forest. Building a random forest 
# involves building 10-100 (usually) decision trees. Each of the decision trees is faster to build than a standard decision tree because 
# of how we do not compare every feature at every split, however given the quantity of decision trees it is often slow to build.

# Similarly, predicting with a Random Forest will be slower than a Decision Tree since we have to do a prediction with each of the 
# 10-100 decision trees in order to get our final prediction.
# Random Forests are not the fastest model, but generally this is not a problem since the computational power of computers is a lot.


# Neural Network Use Cases


# Neural Networks are incredibly popular and powerful machine learning models. They often perform well in cases where we have a lot of features 
# as they automatically do feature engineering without requiring domain knowledge to restructure the features.

# In this module we will be using image data. Since each pixel in the image is a feature, we can have a really large feature set. They are all 
# commonly used in text data as it has a large feature set as well. Voice recognition is another example where neural networks often shine.
# Neural networks often work well without you needing to use domain knowledge to do any feature engineering.

# Biological Neural Network

# A more accurate term for Neural Networks is Artificial Neural Networks (ANN). They were inspired by how biological neural networks work in human brains.

# A brain’s neural network is made up of around 86 billion neurons. The neurons are connected by what are called synapses. There are about 
# 100 trillion synapses in the human brain. The neurons send signals to each other through the synapses.
# We will see in the next lessons how an artificial neural network is defined.


# What's a Neuron?

# An artificial neuron (often called a node) is modeled after a biological neuron. It is a simple object that can take input, do some calculations with the input, and produce an output.

# We visually represent neurons as follows. x1 and x2 are the inputs. Inside the neuron, some computation is done based on x1 and x2 to produce the output y1.

# x1 ->
#         O   -> y1
# x2 -> 

# Neurons can take any number of inputs and can also produce any number of outputs.
# Each neuron is only capable of a small computation, but when working together they become capable of solving large and complicated problems.

# Neuron Computations

# Inside the neuron, to do the computation to produce the output, we first put the inputs into the following equation (just like in logistic regression).

# W1X1 + W2X2 + b

# Recall that x1 and x2 are the inputs. In logistic regression, we referred to the values w1, w2, and b as the coefficients. In neural networks, we refer to w1 and w2 as the weights, and b as the bias.

# We plug this value into what is called an activation function. The above equation can have a result of any real number. The activation function condenses it into a fixed range (often between 0 and 1).

# A commonly used activation function is the sigmoid function, the same function we used in logistic regression. Recall that this function produces a value between 0 and 1. It is defined as follows.


# sigmoid(x) = 1/1+e^-x

# To get the output from the inputs we do the following computation. The weights, w1 and w2, and the bias, b, control what the neuron does. We call these values (w1, w2, b) the parameters. 
# The function f is the activation function (in this case the sigmoid function). The value y is the neuron’s output.

# y = f(W1X1 + W2X2 + b) = 1/1+e^-(W1X1 + W2X2 + b)

# This function can be generalized to have any number of inputs (xi) and thus the corresponding number of weights (wi).


# Activation Functions

# There are three commonly used activation functions: sigmoid (from the previous part), tanh, and ReLU.

# Tanh has a similar form to sigmoid, though ranges from -1 to 1 instead of 0 to 1. Tanh is the hyperbolic tan function and is defined as follows:

# f(x) = tanh(x) = sinh(x)/cosh(x) = e^x - e^-x/e^x + e^-x

# ReLU stands for Rectified Linear Unit. It is the identity function for positive numbers and sends negative numbers to 0.

# Here is the equation and graph.

# ReLU(x) = {0 if x <= 0, x if x > 0}

# Any of these activation functions will work well. Which one to use will depend on specifics of our data. 
# In practice, we figure out which one to use by comparing the performance of different neural networks.

# An Example


# Assume we have a neuron that takes 2 inputs and produces 1 output and whose activation function is the sigmoid. The parameters are:

# Weights (w1, w2) = [0, 1]
# Bias (b) = 2

# If we give the neuron input (1, 2) we get the following calculation.

# w1x1 + w2x2 + b = 0 * 2 + 1x - 2 + 2
#                 = 0 - 2 + 2 = 0
# y = f(w1x1 + w2x2 + b) = 1/(1+e^-4) = 0.9820

# The neuron yields an output of 0.9820.

# Alternatively, if we give the neuron input (2, -2) we get the following calculation.

# w1x1 + w2x2 + b = 0 * 2 + 1 X -2 + 2
#                 = 0 - 2 + 2 = 0
# y = f(w1x1 + w2x2 + b) = 1 / (1 + e^0) = 0.5

# The neuron with this input yields an output of 0.5.
# A neuron by itself does not have much power, but when we build a network of neurons, we can see how powerful they are together.



# Multi-Layer Perceptron

# To create a neural network we combine neurons together so that the outputs of some neurons are inputs of other neurons. 
# We will be working with feed forward neural networks which means that the neurons only send signals in one direction. 
# In particular, we will be working with what is called a Multi-Layer Perceptron (MLP). The neural network has multiple layers which we see depicted below.

# A multi-layer perceptron will always have one input layer, with a neuron (or node) for each input. In the neural network above, there are two inputs and thus two input nodes. 
# It will have one output layer, with a node for each output. Above there is 1 output node for a single output value. 
# It can have any number of hidden layers and each hidden layer can have any number of nodes. Above there is one hidden layer with 5 nodes.

# The nodes in the input layer take a single input value and pass it forward. The nodes in the hidden layers as well as the output layer 
# can take multiple inputs but they always produce a single output. Sometimes the nodes need to pass their output to multiple nodes. 
# In the example above, the nodes in the input layer pass their output to each of the five nodes in the hidden layer.
# A single-layer perceptron is a neural network without any hidden layers. These are rarely used. 
# Most neural networks are multi-layer perceptrons, generally with one or two hidden layers.


# Example Neural Network

# Let’s dive deeper into how this works with an example. A neural network that solves any real problem will be too large to interpret, so we will walk through a simple example.

# We have a neural network with two inputs, a single hidden layer with two nodes and one output. The weights and bias are given in the nodes below. All the nodes use the sigmoid activation function.

# Let’s see what output we get for the input (3,2).

# Here is the output for the first node in the hidden layer.
# h1  = f(0 * x1 + 1 * x2 + 0)
#     = f(0 * 3 + 1 * 2¨+ 0)
#     = f(2)
#     = 1/1 + e^-2
#     = 0.88088

# Here is the output for the second node in the hidden layer.
# h2  = f(1 * x1 + 2 * x2 - 1)
#     = f(1 * 3 + 2 * 2 - 1)
#     = f(6)
#     = 1/1 + e^-6
#     = 0.9975

# Here is the output from the node in the output layer. Note that this node takes the outputs from the hidden layer as input.
# y1  = f(1 * h1 - 1 * h2 + 2)
#     = f(1 * 0.8808 - 1 * 0.9975 + 2)
#     = f(1.8833)
#     = 1 / 1 + e^-l.8833
#     = 0.8680

# Thus for the input (3, 2), the output of the neural network is 0.8680.
# To change how the neural network performs, we can change the weights and bias values.


# More Than 2 Target Values

# A nice benefit of an MLP classifier is that it easily extends to problems that have more than 2 target values. 
# In the previous modules, we have dealt with predicting 0 or 1 (true or false, survived or not, cancerous or not, ). 
# In some cases, we will be choosing among 3 or more possible outputs. A neural network does this naturally. 
# We just need to add more nodes to the output layer. For example, if we are trying to predict if an image is a bird, cat or dog, we will have three output nodes. 
# The first (y1) measures if the image is a bird, the second (y2) measures if the image is a cat, and the third (y3) measures if the image is a dog. 
# The model chooses the output with the highest value.

# Loss


# In order to train a neural network, we need to define a loss function. This is a measure of how far off our neural network is from being perfect. 
# When we train the neural network, we are optimizing a loss function.

# We will use cross entropy as our loss function. This is the same as the likelihood we used in logistic regression but is called by a different name in this context. 
# We calculate the cross entropy as follows.

#                     p   if y = 1
# cross entropy = {
#                     1-p if y = 0

# We multiply together the cross entropy values for all the datapoints.

# Let’s say we have two models to compare on a tiny dataset with 4 datapoints. Here is a table of the true values, the predicted probabilities for model 1 and the predicted probabilities for model 2.

# Target      Model Prediction 1      Model Prediction 2
# 1                   0.6                     0.5
# 1                   0.8                     0.9
# 0                   0.3                     0.1
# 0                   0.4                     0.5


# The cross entropy for model 1 is as follows.
# 0.6 * 0.8 * (1 - 0.3) * (1 - 0.4) = 0.2016

# The cross entropy for model 2 is as follows.
# 0.5 * 0.9 * (1 - 0.1) * (1 - 0.5) = 0.2025

# Cross entropy will be higher the better the model is, thus since model 2 has
# higher cross entropy than model 1, it is the better model.

# Just like we did with the likelihood function in logistic regression, we use the loss function to find the best possible model.



# Backpropagation

# A neural network has a lot of parameters that we can control. There are several coefficients for each node and there can be a lot of nodes! 
# The process for updating these values to converge on the best possible model is quite complicated. The neural network works backwards from 
# the output node iteratively updating the coefficients of the nodes. This process of moving backwards through the neural network is called backpropagation or backprop.

# We won't go through all the details here as it involves calculating partial derivatives, but the idea is that we initialize all the 
# coefficient values and iteratively change the values so that at every iteration we see improvement in the loss function. 
# Eventually we cannot improve the loss function anymore and then we have found our optimal model.
# Before we create a neural network we fix the number of nodes and number of layers. Then we use backprop to iteratively 
# update all the coefficient values until we converge on an optimal neural network.



# Creating Artificial Dataset

# Sometimes in order to test models, it is helpful to create an artificial dataset. We can create a dataset of the size and complexity needed. 
# Thus we can make a dataset that is easier to work with than a real life dataset. This can help us understand how models work before we apply them to messy real world data.

# We will use the make_classification function in scikit-learn. It generates a feature matrix X and target array y. We will give it these parameters:

# • n_samples: number of datapoints
# • n_features: number of features
# • n_informative: number of informative features
# • n_redundant: number of redundant features
# • random_state: random state to guarantee same result every time

# You can look at the full documentation to see other parameters that you can tweak to change the result.

# Here is the code to generate a dataset.

from sklearn.datasets import make_classification
X, y = make_classification(n_features = 2, n_redundant = 0, n_informative = 2, random_state=3)

from matplotlib import pyplot as plt
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], s=100, edgecolors='k')
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], s=100, edgecolors='k', marker='^')
plt.show()

# Scikit-learn has a couple other functions besides make_classification for making classification datasets with different properties. 
# Look at make_circles and make_moons if you want to play around with more artificial datasets.



# MLPClassifier

# Scikit-learn has an MLPClassifier class which is a multi-layer perceptron for classification. We can import the class from scikit-learn, 
# create an MLPClassifier object and use the fit method to train.

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_features = 2, n_redundant = 0, n_informative = 2, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 3)
mlp = MLPClassifier()
mlp.fit(X_train, y_train)

# You will notice that we get a ConvergenceWarning. 
# This means that the neural network needs more iterations to converge on the optimal coefficients. 
# The default number of iterations is 200. Let’s up this value to 1000.

mlp = MLPClassifier(max_iter=1000)

# Now when we run this code, the neural network will converge. We can now use the score method to calculate the accuracy on the test set.

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 3)
mlp = MLPClassifier(max_iter=1000)
mlp.fit(X_train, y_train)
print("Accuracy: ", mlp.score(X_test, y_test))

# Neural networks are incredibly complicated, but scikit-learn makes them very approachable to use!


# Parameters for MLPClassifier

# There are a couple of parameters that you may find yourself needing to change in the MLPClassifier.

# You can configure the number of hidden layers and how many nodes in each layer. The default MLPClassifier 
# will have a single hidden layer of 100 nodes. This often works really well, but we can experiment with different values. 
# This will create an MLPCLassifier with two hidden layers, one of 100 nodes and one of 50 nodes.

mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 50))

# We saw max_iter in the previous part. This is the number of iterations. In general, the more data you have, 
# the fewer iterations you need to converge. 
# If the value is too large, it will take too long to run the code. If the value is too small, 
# the neural network will not converge on the optimal solution.

# We also sometimes need to change alpha, which is the step size. This is how much the neural network changes the coefficients 
# at each iteration. If the value is too small, you may never converge on the optimal solution. If the value is too large, 
# you may miss the optimal solution. Initially you can leave this at the default. The default value of alpha is 0.0001. 
# Note that decreasing alpha often requires an increase in max_iter.

# Sometimes you will want to change the solver. This is what algorithm is used to find the optimal solution. 
# All the solvers will work, but you may find for your dataset that a different solver finds the optimal solution faster. 
# The options for solver are 'lbfgs', 'sgd' and 'adam'.

# Run this code in the playground and try changing the parameters for the MLPClassifier. The code uses a random_state 
# to ensure that every time you run the code with the same parameters you will get the same output.


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 3)
mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 50), alpha=0.0001, solver = 'adam', random_state = 3)
mlp.fit(X_train, y_train)
print("Accuracy: ", mlp.score(X_test, y_test))

# If you look at the docs, you can read about several more parameters that you can tune in the neural network.


# The MNIST Dataset

# In this lesson we will be working with a new dataset, the MNIST database of handwritten digits. 
# NIST is the National Institute of Standards and Technology and the M stands for Modified.

# This is a database of images of handwritten digits. We will build a classifier to determine which digit is in the image.

# We will start with the version of the MNIST dataset that is built into scikit-learn. This has the images with only 8 by 8 pixels, 
# so they are blurry.

# Here are a couple example images:

# In scikit-learn we can load the dataset using the load_digits function. To simplify the problem, 
# we will initially only be working with two digits (0 and 1), so we use the n_class parameter to limit the number of target values to 2.

from sklearn.datasets import load_digits
X, y = load_digits(n_class=2, return_X_y=True)

# We can see the dimensions of X and y and what the values look like as follows.

print(X.shape, y.shape)
print(X[0])
print(y[0])

# We see that we have 300 datapoints and each datapoint has 64 features. We have 64 features because the 
# image is 8 x 8 pixels and we have 1 feature per pixel. The value is on a grayscale where 0 is black and 16 is white.

# To get a more intuitive view of the datapoint, reshape the array to be 8x8.

print(X[0].reshape(8, 8))

# We can see that this is a 0, though we will see in the next part that we can draw the image more clearly.

from sklearn.datasets import load_digits
X, y = load_digits(n_class=2, return_X_y=True)
print(X.shape, y.shape)
print(X[0])
print(y[0])
print(X[0].reshape(8, 8))

# There are different versions of this dataset with more pixels and with colors (not grayscale). 
# We will see that even with these simplified images, we can build a good classifier.



# Drawing the Digits

# You can build a model without ever looking at a visual representation of the images, but it can sometimes be helpful to draw the image.

# We use the matplotlib function matshow to draw the image. The cmap parameter is used to indicate that the image should be in a grayscale rather than colored.

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

X, y = load_digits(n_class=2, return_X_y=True)
plt.matshow(X[0].reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())  # remove x tick marks
plt.yticks(())  # remove y tick marks
plt.show()

# You can see that with only 64 pixels the image is very pixelated. Even with these blurry images we can build an excellent model.

# MLP for MNIST Dataset


# Now let’s use the MLPClassifier to build a model for the MNIST dataset.

# We will do a train/test split and train an MLPClassifier on the training set.

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
mlp = MLPClassifier()
mlp.fit(X_train, y_train)

# We do not get a warning, so the default number of iterations is adequate in this case.

# Let’s look at how the model predicts the first datapoint in the test set. We use matplotlib to draw the images and then show the model’s prediction.

x = X_test[0]
plt.matshow(x.reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.show()
print(mlp.predict([x]))
# 0

# We can see that this is a 0 and that our model correctly predicts 0.

# Similarly, let’s look at the second datapoint.

x = X_test[1]
plt.matshow(x.reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.show()
print(mlp.predict([x]))
# 1

# This is clearly a 1 and our model again gets the correct prediction.

# We can also see the model’s accuracy on the entire test set.

print(mlp.score(X_test, y_test))

# We can also see the model's accuracy on the entire test set. Thus our model gets 100% accuracy.
# 0 and 1 are two of the easier digits to distinguish, but we will see that the model can also perform well with distinguishing harder examples.



# Classifying all 10 Digits

# Since neural networks easily generalize to handle multiple outputs, we can just use the same code to build a classifier to 
# distinguish between all ten digits.

# This time when we load the digits, we do not limit the number of classes.

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
mlp = MLPClassifier(random_state=2)
mlp.fit(X_train, y_train)

print(mlp.score(X_test, y_test))

# So we got 96% of the datapoints in the test set correct. Let’s look at the ones we got incorrect. We use a numpy mask to pull 
# out just the datapoints we got incorrect. We pull the x values, the true y value as well as the predicted value.

y_pred = mlp.predict(X_test)
incorrect = X_test[y_pred != y_test]
incorrect_true = y_test[y_pred != y_test]
incorrect_pred = y_pred[y_pred != y_test]

# Let’s look at the first image that we got wrong and what our prediction was.

j = 0
plt.matshow(incorrect[j].reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.show()
print("true value:", incorrect_true[j])
print("predicted value:", incorrect_pred[j])

# true value: 4
# predicted value: 9

# You can see from looking at the image that a human might also be confused. It is not obvious whether it is a 4 or a 9.

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
mlp = MLPClassifier(random_state=2)
mlp.fit(X_train, y_train)

print(mlp.score(X_test, y_test))

y_pred = mlp.predict(X_test)
incorrect = X_test[y_pred != y_test]
incorrect_true = y_test[y_pred != y_test]
incorrect_pred = y_pred[y_pred != y_test]

j = 0
plt.matshow(incorrect[j].reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.show()
print("true value:", incorrect_true[j])
print("predicted value:", incorrect_pred[j])

# You can modify the code to see all of the datapoints the model predicted incorrectly.



# Open ML


# For this lesson, we will use a more granular version of the MNIST dataset. Instead of using the version in 
# scikit-learn which has 64 pixel images, we will use a version from Open ML that has 784 pixels (28 x 28).

# Open ML (www.openml.org) has a database of large datasets that can be used for a variety of machine learning problems. 
# Scikit-learn has a function fetch_openml for directly downloading datasets from the Open ML database.

# Use the following code to get our dataset.

from sklearn.datasets import fetch_openml
import numpy as np
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# We can briefly look at the shape of the arrays, the range of the features values, and the first few values of the 
# target array to better understand the dataset.

print(X.shape, y.shape)
print(np.min(X), np.max(X))
print(y[0:5])

# We can see that we have 70,000 datapoints with 784 features. The feature values range from 0 to 255 
# (which we interpret on a gray scale with 0 being white and 255 being black). The target values are the numbers 0-9. 
# Note that the target values are stored as strings and not integers.

# For our example, we will be using only the digits 0-3, so we can use the following code to segment out that portion of the dataset.

X5 = X[y <= '3']
y5 = y[y <= '3']

# We will be modifying some of the default parameters in the MLPClassifier to build the model. 
# Since our goal will be to visualize the weights of the hidden layer, we will use only 6 nodes in the hidden layer so 
# that we can look at all of them. We will use 'sgd' (stochastic gradient descent) as our solver which requires us to decrease 
# alpha (the learning rate).

mlp=MLPClassifier(
  hidden_layer_sizes=(6,), 
  max_iter=200, alpha=1e-4,
  solver='sgd', random_state=2)

mlp.fit(X5, y5)

# If we run this code we will see that it converges.
# Since this dataset is quite large, you will want to work with it on your computer rather than the code playground.



# MLPClassifier Coefficients

# The MLPClassifier stores the coefficients in the coefs_ attribute. Let’s see what it looks like.

print(mlp.coefs_)

# First we see that it is a list with two elements.

print(len(mlp.coefs_))

# The two elements in the list correspond to the two layers: the hidden layer and the output layer. We have an array of coefficients 
# for each of these layers. Let’s look at the shape of the coefficients for the hidden layer.

print(mlp.coefs_[0].shape)
# Output:
# (784, 6)

# We see that we have a 2-dimensional array of size 784 x 6. There are 6 nodes and 784 input values feeding into each node, and we 
# have a weight for each of these connections.
# In order to interpret the values, we will need to use a visual representation.


# Visualizing the Hidden Layer

# To get a better understanding of what the neural network is doing, we can visualize the weights of the hidden layer to get some insight 
# into what each node is doing.

# We will use the matshow function from matplotlib again to draw the images. In matplotlib we can use the subplots function to create 
# multiple plots within a single plot.

fig, axes = plt.subplots(2, 3, figsize=(5, 4))
for i, ax in enumerate(axes.ravel()):
    coef = mlp.coefs_[0][:, i]
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(i + 1)
plt.show()


from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

mlp=MLPClassifier(
  hidden_layer_sizes=(6,), 
  max_iter=200, alpha=1e-4,
  solver='sgd', random_state=2)

mlp.fit(X_train, y_train)


fig, axes = plt.subplots(2, 3, figsize=(5, 4))
for i, ax in enumerate(axes.ravel()):
    coef = mlp.coefs_[0][:, i]
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(i + 1)
plt.show()

# You can see that nodes 4 and 6 are determining if the digit is a 3. 
# Node 1 is determining if the digit is a 0 or a 2 since you can see both of those values in the image. 
# Not every hidden node will have an obvious use.
# If you change the random state in the MLPClassifier, you will likely get different results. 
# There are many equivalently optimal neural networks that work differently.


# Interpretability

# While we can visualize the nodes in the hidden layer to understand on a high level what the neural network is doing, 
# it is impossible to answer the question "Why did datapoint x get prediction y?" Since there are so many nodes, each with 
# their own coefficients, it is not feasible to get a simple explanation of what the neural network is doing. 
# This makes it a difficult model to interpret and use in certain business use cases.
# Neural Networks are not a good option for interpretability.

# Computation

# Neural networks can take a decent amount of time to train. Each node has its own coefficients and to train they are iteratively 
# updated, so this can be time consuming. However, they are parallelizable, so it is possible to throw computer power at them to 
# make them train faster.
# Once they are built, neural networks are not slow to make predictions, however, they are not as fast as some of the other models.


# Performance

# The main draw to neural networks is their performance. On many problems, their performance simply cannot be beat by other models. 
# They can take some tuning of parameters to find the optimal performance, but they benefit from needing minimal feature engineering 
# prior to building the model.

# A lot of simpler problems, you can achieve equivalent performance with a simpler model like logistic regression, but with 
# large unstructured datasets, neural networks outperform other models.
# The key advantage of neural networks is their performance capabilities.






















































































































# %%
# %%
