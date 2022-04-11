#Este es un archivo sin ningún tipo de lógica, sólo busca anotar las variables y funciones que voy aprendiendo
#en SoloLearn, para tener un respaldo y una guía de donde puedo acudir sobre lo que he aprendido.
#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

# %%