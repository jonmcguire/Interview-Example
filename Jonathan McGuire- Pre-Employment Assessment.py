"""
@author: Jonathan McGuire
"""

#Package Import
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score
from sklearn.svm import SVC


#Load data
train=pd.read_csv('exercise_02_train.csv')
test=pd.read_csv('exercise_02_test.csv')

#combine both
traintest=pd.concat([train,test])

#Split target from data
y=train['y']
traintest=traintest.drop('y',axis=1)



#Part 1-Data Cleaning and Analysis
#Check types, number of missing values and head
traintesthead=traintest.head()
traintestisnull=traintest.isnull().sum()
vartypes=traintest.dtypes

##Numerics- Remove $ from x41, and % from x45
traintest['x41']=traintest['x41'].str[1:].astype("float64")
traintest['x45']=traintest['x45'].str[:-1].astype("float64")
###Columns x41 and x45 were categorical when they should be numeric


##Categorical- Replace typos
traintest['x35']=traintest['x35'].replace({"wed":"wednesday", "thur": "thursday", "thurday": "thursday", "fri": "friday"})
traintest['x68']=traintest['x68'].str.lower().str[:3].replace({"dev":"dec"})
traintest['x93']=traintest['x93'].replace({"euorpe":"europe"})

desc=traintest.describe()
desc=desc.transpose()
desc[desc['count']<40000].sort_values(by='count')

##Distribution of target variable
y.value_counts() #target variable is not significantly unbalanced, no action neccesary


##Seperate Numeric and Categorical Columns
numcol=traintest.select_dtypes(exclude='object')
catcol=traintest.select_dtypes(include='object')

###Check Categorical columns counts
for col in catcol:
    print(traintest[col].str.lower().value_counts())


##Replace nan with median
numcol=numcol.fillna(numcol.median())

for col in catcol:
    print("Column Mode: ",catcol[col].mode()[0])
    catcol[col]=catcol[col].fillna(catcol[col].mode()[0])


##Combine catcol and numcol into traintest
traintest=pd.concat([numcol,catcol],axis=1)

##Seperate train from test, 40000 row and categories from numerics
X=traintest.iloc[:40000,:] 
X_test=traintest.iloc[40000:,:]



numcoltrain=X.select_dtypes(exclude='object')
catcoltrain=X.select_dtypes(include='object')

numcoltest=X_test.select_dtypes(exclude='object')
catcoltest=X_test.select_dtypes(include='object')


##Standardize scale of numerics
scaler= StandardScaler()
numcoltrain = pd.DataFrame(scaler.fit_transform(numcoltrain))

X=pd.concat([numcoltrain,catcoltrain],axis=1)

scaler= StandardScaler()
numcoltest = pd.DataFrame(scaler.fit_transform(numcoltest))

X_test=pd.concat([numcoltest,catcoltest],axis=1)




#Part 2-Model Building and Part 3-Generate Predictions
##Since the target variable is 1 or 0, the problem is a classification problem
##Logistic Regression is a good baseline model, because it is interpretable and simpler
##Support Vector Machines is a more resource intensive model, that may provide better metrics

##create dummy variables for categorical variables
X = pd.get_dummies(X, columns=['x34','x35','x68','x93'])
X_test = pd.get_dummies(X_test, columns=['x34','x35','x68','x93'])


##split into training and validation set
X_train, X_valid, y_train, y_valid =train_test_split(X, y, test_size=0.2, random_state=0)


##function to run models
def run_model(model, X_train, X_valid, y_train, y_valid,X, y, X_test ):
    ##fit model
    model.fit(X_train,y_train)
    y_pred=model.predict(X_valid)
    
    ##model evaluation
    print("Model: ",model)
    print("Training Accuracy: ", round(model.score(X_train, y_train) * 100, 2), "%")
    print("Validation Accuracy: ", round(model.score(X_valid, y_valid) * 100, 2), "%")
    print("AUC: ",roc_auc_score(y_valid, y_pred))
    
    print("Cross Validation Accuracy: ",round((cross_val_score(model, X, y, cv=10).mean()*100),2),"%")
    
    print(classification_report(y_valid, y_pred))
    print("Validation Confusion Matrix: ","\n",confusion_matrix(y_valid, y_pred))
    
    return model.predict_proba(X_test)
    

##Logistic Regression
logmodpred=run_model(LogisticRegression(solver='lbfgs',max_iter=10000),X_train, X_valid, y_train, y_valid,X, y,X_test)
logmodpred=pd.DataFrame(logmodpred).drop([0], axis = 1)
logmodpred.to_csv("results1.csv",index=False,header=False)

##Training Accuracy:  89.2 %
##Validation Accuracy:  88.79 %
##AUC:  0.7780754517323819
###model performs average in class seperation

##Cross Validation Accuracy:  89.07 %
##No sign of significant overfitting

#              precision    recall  f1-score   support

#           0       0.90      0.96      0.93      6379
#           1       0.80      0.59      0.68      1621

#    accuracy                           0.89      8000
#   macro avg       0.85      0.78      0.81      8000
#weighted avg       0.88      0.89      0.88      8000

#[[6141  238]
# [ 659  962]]
##Predicting 0 is significantly more accurate than predicting 1

##Support Vector Machine
svmmodpred=run_model(SVC(probability=True),X_train, X_valid, y_train, y_valid,X, y, X_test)
svmmodpred=pd.DataFrame(svmmodpred).drop([0], axis = 1)
svmmodpred.to_csv("results2.csv",index=False,header=False)

##Training Accuracy:  99.38 %
##Validation Accuracy:  98.52 %
##AUC:  0.9663635469522867
###model performs very high in class seperation

##Cross Validation Accuracy:  98.79 %
##No sign of significant overfitting

#              precision    recall  f1-score   support
#
#           0       0.98      1.00      0.99      6379
#           1       0.99      0.93      0.96      1621

#    accuracy                           0.99      8000
#   macro avg       0.99      0.97      0.98      8000
#weighted avg       0.99      0.99      0.99      8000

#[[6367   12]
# [ 106 1515]]

