import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

myData = pd.read_csv(r"D:\Bastian\Dev\SoloLearn\myCSV.csv")


feature_names = ['Sex', 'Age', 'Siblings', 'Father', 'Mother', 'Family Income', 'Debts']

X = myData[feature_names].values

# df = myData['Beca']
# def df(md):
#     a = 0
#     if a <= 5 :
#         if ['Father'] == 0:
#             a =+ 1
#         elif ['Mother'] == 0:
#             a =+ 1
#         elif ['Debts'] == 1:
#             a =+ 1
#     if a > 5 :
#         a = 1
#     return a
               
y = myData['isBully'].values

X_train, X_test, y_train, y_test = train_test_split(X, y)

rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)

# n_estimators = 10
# param_grid = {
#     'n_estimators': n_estimators,
# }
# rf = RandomForestClassifier()
# gs = GridSearchCV(rf, param_grid, cv=5)
# gs.fit(X, y)
# scores = gs.cv_results_['mean_test_score']
# import matplotlib.pyplot as plt

# scores = gs.cv_results_['mean_test_score']
# plt.plot(n_estimators, scores)
# plt.xlabel("n_estimators")
# plt.ylabel("accuracy")
# plt.xlim(0, 100)
# plt.ylim(0.9, 1)
# plt.show()

print(rf.predict(X_test))

print(rf.score(X_test, y_test))
print("Predicción: ")
print(rf.predict([[0,13,2,0,1,700000,1]]))



#%%














# %%
