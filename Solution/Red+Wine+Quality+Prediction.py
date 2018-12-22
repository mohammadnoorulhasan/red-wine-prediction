# Created on Fri Aug 03 11:00:00 2018
# Problem Number    :   1 
# Problem title     :   Wine Quality Prediction
# Author            :   Mohammad Noor Ul Hasan
# Last Edit         :   Mon Aug 7 2018

# Language          :   Python 3
# Module Used       :   pandas, matplotlib, seaborn, classification report, confusion matrix
# Algorithm Used    :   SVM , KNN, GridSearchCV, StandardScaler 
# Data Structures   :   Dictionary, DataFrames, List


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')

from sklearn.svm import SVC  #Support Vector Classifier (SVM) algorithm module
from sklearn.neighbors import KNeighborsClassifier # K nearest neighbour (KNN) algorithm module
from sklearn.model_selection import GridSearchCV  # For finding best parameters of SVM
from sklearn.preprocessing import StandardScaler # To standardizethe data
from sklearn.metrics import classification_report # To generate Classification Report
from sklearn.metrics import confusion_matrix # To generate Confusion Matrix



data1 = pd.read_csv("../Problem/Training.csv") # Loading Training Data 
data2 = pd.read_csv("../Problem/Testing.csv")  # Loading Testing Data

# Initialize the Standard Scaler which Standardize the data
scaler = StandardScaler()

# Calculating Mean, Standard Deviation of loading Data
scaler.fit(data1) 

# Transforming the data into its standardize form with Mean, Standard deviation of Training Data 
train = pd.DataFrame(scaler.transform(data1),columns=data1.columns)
test = pd.DataFrame(scaler.transform(data2),columns=data2.columns)

# Spliting Independent and dependent variable
# Independent Variable
X_train = train.drop("quality",axis = 1) # Of Training
X_test = test.drop("quality",axis = 1)   # Of Testing

# Independent Variable
y_train = data1.quality
y_test = data2.quality



# Searching for best kernal, gamma, C of SVM
#uncomment the bottom line to search 
#comment these line because it slow downs the process
#param = {
#    'C': [0.1,0.2,0.3,0.8,0.9,1,1.1,1.2,1.3,1.4],
#    'kernel':['linear', 'rbf'],
#    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
#}
#grid_svc = GridSearchCV(SVC(), param_grid=param, scoring='accuracy', cv=10)
#grid_svc.fit(X_train, y_train)
#print(grid_svc.best_params_)
#uncomment till here



# Initializing the SVC model
svm_model = SVC(kernel="rbf",gamma=0.1,C = 1.2)
# Providing Training to the model with Training data 
svm_model.fit(X_train, y_train)

# Predicting the output of testing data and saving it in y_predict_SVM
y_predict_SVM = svm_model.predict(X_test)

# Confidence of the model
print( 'SVM confidence with training data:', svm_model.score(X_train, y_train))
print( 'SVM confidence with testing data:', svm_model.score(X_test, y_test))

# Confusion Matrix for SVM model
confusion_matrix_SVM = confusion_matrix(y_test,y_predict_SVM)
cols = list(sorted(y_train.unique()))
confusion_matrix_SVM_df =  pd.DataFrame(confusion_matrix_SVM,columns=["predicted "+str(i) for i in cols],index=[i for i in cols])
confusion_matrix_SVM_df["total"] = y_test.value_counts()

# Classification report of SVM
classification_report_SVM = classification_report(y_test, y_predict_SVM)



# Initializing the KNN model
knn_model = KNeighborsClassifier(n_neighbors = 18)

# Providing Training to the model with Training data 
knn_model.fit(X_train, y_train)

# Predicting the output of testing data and saving it in y_predict_KNN
y_predict_KNN = knn_model.predict(X_test)
print( 'KNN confidence with training data :', knn_model.score(X_train, y_train))
print( 'KNN confidence with testing data :', knn_model.score(X_test, y_test))

# Confusion Matrix for KNN model
confusion_matrix_KNN = confusion_matrix(y_test,y_predict_KNN)
cols = list(sorted(y_train.unique()))
confusion_matrix_KNN_df = pd.DataFrame(confusion_matrix_KNN,columns=["predicted "+str(i) for i in cols],index=[i for i in cols])
confusion_matrix_KNN_df["total"] = y_test.value_counts()

# Classification report of KNN model
classification_report_KNN = classification_report(y_test, y_predict_KNN)


# # Ploting of KNN Confusion matrix

plt.figure(figsize = (10,7))
print("KNN Model Confusion Matrix")
sns.heatmap(confusion_matrix_KNN_df, annot=True)
plt.show()



y = []
for index in confusion_matrix_KNN_df.index:
    y.append(confusion_matrix_KNN_df.loc[index,"predicted "+str(index)])
    x = confusion_matrix_KNN_df.loc[:,"predicted "+str(index)]
    plt.bar(cols,x)
    plt.title('KNN actual V/s prediction for '+str(index))
    plt.xlabel("Predicted "+str(index)+" as ")
    plt.ylabel("count")
    plt.show()
plt.title('Actual output')
plt.xlabel("Frequency Term")
plt.ylabel("Frequency")
plt.bar(cols,y)

plt.show()
print(confusion_matrix_KNN_df)


# # Ploting of SVM Confusion Matrix



plt.figure(figsize = (10,7))

print("SVM Model Confusion Matrix")
sns.heatmap(confusion_matrix_SVM_df, annot=True)
plt.show()



y = []
for index in confusion_matrix_SVM_df.index:
    y.append(confusion_matrix_SVM_df.loc[index,"predicted "+str(index)])
    x = confusion_matrix_SVM_df.loc[:,"predicted "+str(index)]
    plt.bar(cols,x)
    plt.title('SVM actual V/s prediction for '+str(index))
    plt.xlabel("Predicted "+str(index)+" as ")
    plt.ylabel("count")
    plt.show()
plt.bar(cols,y)
plt.title('Actual output')
plt.xlabel("Frequency Term")
plt.ylabel("Frequency")
plt.show()
print(confusion_matrix_SVM_df)


# # Classification Report Of KNN


print("KNN Model Classification Report ")
print(classification_report_KNN)


# # Classification Report Of SVM


print("SVM Model Classification Report ")
print(classification_report_SVM)



#Comparision between confidence of KNN and SVM algorithm
svm_score = svm_model.score(X_test, y_test)
knn_score = knn_model.score(X_test, y_test)
print('Confidence of SVM  with testing data:', svm_score)
print('Confidence of KNN  with testing data:', knn_score)

if(knn_score > svm_score):
    model = knn_model
else:
    model = svm_model


# # Final Model is


print(model)



prediction = model.predict(X_test)
prediction_remark = prediction == y_test
model_output = {"actual_quality":y_test,"predicted_quality":prediction,"prediction_remark":prediction_remark}
final_output = pd.DataFrame(model_output)
final_output.to_csv("output.csv")

print("predicted output is save in 'output.csv' file")
