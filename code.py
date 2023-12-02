#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import openpyxl
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter


# In[2]:


dataSourceExcel = 'Data for ML.xlsx'


# In[3]:


masterDF = pd.read_excel(dataSourceExcel)


# In[4]:


masterDF.head()


# In[5]:


masterDF.drop('ID', axis=1, inplace = True)


# In[6]:


masterDF.head()


# In[7]:


masterDF.shape


# In[8]:


#collect all the titles into a string
allTitlesList = masterDF['Title'].values.tolist()


# In[9]:


allTitleMasterText = ''.join(allTitlesList)


# In[10]:


allTitleMasterText


# In[11]:


import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

stopwords.words('english')


# In[12]:


nltk.download('punkt')


# In[13]:


titleTokens = word_tokenize(allTitleMasterText)


# In[14]:


print(len(titleTokens))


# In[15]:


print(stopwords.words('english'))


# In[17]:


tokens_without_sw = [word for word in titleTokens if not word in stopwords.words()]


# In[ ]:


print(len(tokens_without_sw))


# In[16]:


#remove additonal tokens
discardTokens = ['(', ')', '[', ']',',',':', '-1', '-2', '-3', '-4', 'a', 'an', 'the', 'A', 'An', 'The', '1', '2', 'Is', 'Are', 'In', '*', '-']


# In[17]:


tokens_without_sw_dt = [tok for tok in titleTokens if tok not in discardTokens]


# In[18]:


countDict = Counter(tokens_without_sw_dt)


# In[19]:


countDict


# In[20]:


print(len(countDict))


# In[21]:


countValues = countDict.values()
countValuesDict = Counter(countValues)
xVals = countValuesDict.keys()
yVals = countValuesDict.values()


# In[22]:


plt.scatter(xVals, yVals)
plt.xlim(1,20)
plt.ylim(1,500)
plt.show()


# In[23]:


selectedKeys = list()
for key in countDict:
    if countDict[key]>=10:
        selectedKeys.append(key)
print(len(selectedKeys))


# In[24]:


titleFeatsList = list()
fi = 1
for key in selectedKeys:
    ftname = 'title-'+str(fi)+'-'+key
    titleFeatsList.append(ftname)
    fi = fi + 1
print(titleFeatsList)


# In[25]:


#insert the features based on title strings
ft = selectedKeys[1]
ft


# In[26]:


masterDF['Title'].values[0]


# In[27]:


allRows = list()
for ri in range(2333):
    titleStr = masterDF['Title'].values[ri]
    ftList = list()
    for selFeat in selectedKeys:
        ftList.append(titleStr.count(selFeat))
    print(ftList)
    allRows.append(ftList)    


# In[28]:


for si in range(len(selectedKeys)):
    selFeat = selectedKeys[si]
    ftName = titleFeatsList[si]
    print('ft= ', selFeat)
    ftVals = [ss[si] for ss in allRows]
    print(len(ftVals))
    masterDF[ftName] = ftVals


# In[29]:


masterDF.head()


# In[30]:


newMasterDF = pd.DataFrame(masterDF[titleFeatsList])


# In[31]:


newMasterDF.shape


# In[32]:


newMasterDF['Label'] = masterDF['To Predict: Disease Name']


# In[33]:


newMasterDF.shape


# # ready to apply basic ML
# 

# In[34]:


train=newMasterDF.sample(frac=0.8,random_state=200)
test=newMasterDF.drop(train.index)


# In[35]:


xTrain = train[titleFeatsList]
ytrain = train['Label']


# In[36]:


clf = DecisionTreeClassifier(max_depth =13, random_state = 42)

clf.fit(xTrain, ytrain)


# In[37]:


xTest = test[titleFeatsList]
ytest = test['Label']


# In[38]:


pred_dtree = clf.predict(xTest)


# In[51]:


from sklearn.metrics import confusion_matrix
print("Accuracy:",metrics.accuracy_score(ytest, pred_dtree))

conf_matrix = confusion_matrix(ytest, pred_dtree)
print("Confusion Matrix:")
print(conf_matrix)

num_unique_classes = masterDF['To Predict: Disease Name'].nunique()
print("Number of Unique Classes:", num_unique_classes)


print(masterDF.groupby('To Predict: Disease Name').size())

import pandas as pd

unique_counts = ytrain.value_counts()
print("Count of each unique value in 'To Predict: Disease Name' column:")
print(unique_counts)

unique_counts = ytest.value_counts()
print("Count of each unique value in 'To Predict: Disease Name' column:")
print(unique_counts)


# In[52]:


#display of results

sampleDF = masterDF.sample(n=10)


# In[53]:


sampleDF.shape


# In[54]:


predictedList = list()
for ri in range(2333):
    row = masterDF.iloc[ri][titleFeatsList]
    pred = clf.predict([row])
    print(pred)
    predictedList.append(pred)


# In[72]:


predictedList


# In[73]:


masterDF['PredictedDisease'] = predictedList


# In[48]:


reportFname = '../PredictedDisease.xlsx'
masterDF.to_excel(reportFname)


# In[49]:


import pickle


# In[51]:


# save the model to disk
filename = '../InnomagineTitleRFTrainedClassifierDump.sav'
pickle.dump(clf, open(filename, 'wb'))
 


# In[53]:


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(xTest, ytest)
print(result)


# In[55]:


newMasterDF['Label'] = masterDF['To Predict: Study type']


# In[56]:


newMasterDF.shape


# In[57]:


train=newMasterDF.sample(frac=0.8,random_state=200)
test=newMasterDF.drop(train.index)


# In[58]:


xTrain = train[titleFeatsList]
ytrain = train['Label']


# In[59]:


clf = DecisionTreeClassifier(max_depth =13, random_state = 42)
clf.fit(xTrain, ytrain)


# In[60]:


xTest = test[titleFeatsList]
ytest = test['Label']


# In[61]:


pred_dtree = clf.predict(xTest)


# In[63]:


print("Accuracy:",metrics.accuracy_score(ytest, pred_dtree))

conf_matrix = confusion_matrix(ytest, pred_dtree)
print("Confusion Matrix:")
print(conf_matrix)

num_unique_classes = masterDF['To Predict: Study type'].nunique()
print("Number of Unique Classes:", num_unique_classes)


print(masterDF.groupby('To Predict: Study type').size())

unique_counts = ytrain.value_counts()
print("Count of each unique Training value in 'To Predict: Disease Name' column:")
print(unique_counts)

unique_counts = ytest.value_counts()
print("Count of each unique Testing value in 'To Predict: Disease Name' column:")
print(unique_counts)


# In[82]:


#display of results

sampleDF = masterDF.sample(n=10)


# In[83]:


sampleDF.shape


# In[84]:


predictedList = list()
for ri in range(2333):
    row = masterDF.iloc[ri][titleFeatsList]
    pred = clf.predict([row])
    print(pred)
    predictedList.append(pred)


# In[85]:


predictedList


# In[86]:


masterDF['PredictedStudy'] = predictedList


# In[70]:


reportFname = '../PredictedStudy.xlsx'
masterDF.to_excel(reportFname)


# In[71]:


import pickle
# save the model to disk
filename = '../InnomagineTitleRFTrainedClassifierDump.sav'
pickle.dump(clf, open(filename, 'wb'))
 


# In[72]:


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(xTest, ytest)
print(result)


# In[67]:


import numpy as np
newMasterDF['Label'] = masterDF['To Predict: Intervention']
newMasterDF.shape
train=newMasterDF.sample(frac=0.8,random_state=200)
test=newMasterDF.drop(train.index)
xTrain = train[titleFeatsList]
ytrain = train['Label']
clf = DecisionTreeClassifier(max_depth =13, random_state = 42)
clf.fit(xTrain, ytrain)
xTest = test[titleFeatsList]
ytest = test['Label']
pred_dtree = clf.predict(xTest)
print("Accuracy:",metrics.accuracy_score(ytest, pred_dtree))
#display of results

conf_matrix = confusion_matrix(ytest, pred_dtree)
print("Confusion Matrix:")
dimensions = np.shape(conf_matrix)
rows, columns = dimensions


for i in range(rows):
    print(conf_matrix[i])

num_unique_classes = masterDF['To Predict: Intervention'].nunique()
print("Number of Unique Classes:", num_unique_classes)



print(masterDF.groupby('To Predict: Intervention').size())

unique_counts = ytrain.value_counts()
print("Count of each unique Training value in 'To Predict: Disease Name' column:")
print(unique_counts)

unique_counts = ytest.value_counts()
print("Count of each unique Testing value in 'To Predict: Disease Name' column:")
print(unique_counts)


sampleDF = masterDF.sample(n=10)
sampleDF.shape
predictedList = list()
for ri in range(2333):
    row = masterDF.iloc[ri][titleFeatsList]
    pred = clf.predict([row])
    print(pred)
    predictedList.append(pred)
predictedList
len(predictedList)
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(xTest, ytest)
print(result)


# In[ ]:


#random forest


# In[65]:


# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# Step 2: Load and preprocess your dataset from an xlsx file
# Replace 'your_data.xlsx' with your dataset's filename
data = pd.read_excel('Data for ML.xlsx')

# Step 3: Vectorize the 'title' text data
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf_vectorizer.fit_transform(data['Title'])

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['To Predict: Study type'], test_size=0.2, random_state=42)

# Step 5: Train a Random Forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = random_forest_classifier.predict(X_test)

# Step 7: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))

# Print a classification report for more details
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

num_unique_classes = data['To Predict: Study type'].nunique()
print("Number of Unique Classes:", num_unique_classes)

unique_counts = ytrain.value_counts()
print("Count of each unique Training value in 'To Predict: Study type' column:")
print(unique_counts)

unique_counts = ytest.value_counts()
print("Count of each unique Testing value in 'To Predict: Study type' column:")
print(unique_counts)


# In[88]:


from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

# Calculate precision and recall
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print(classification_report(y_test, y_pred))


# In[97]:


# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# Step 2: Load and preprocess your dataset from an xlsx file
# Replace 'your_data.xlsx' with your dataset's filename
data = pd.read_excel('Data for ML.xlsx')

# Step 3: Vectorize the 'title' text data
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf_vectorizer.fit_transform(data['Title'])

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['To Predict: Disease Name'], test_size=0.2, random_state=42)

# Step 5: Train a Random Forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = random_forest_classifier.predict(X_test)

# Step 7: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))

# Print a classification report for more details
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

num_unique_classes = data['To Predict: Disease Name'].nunique()
print("Number of Unique Classes:", num_unique_classes)


# In[87]:


from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

# Calculate precision and recall
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print(classification_report(y_test, y_pred))


# In[98]:


# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# Step 2: Load and preprocess your dataset from an xlsx file
# Replace 'your_data.xlsx' with your dataset's filename
data = pd.read_excel('Data for ML.xlsx')

# Step 3: Vectorize the 'title' text data
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf_vectorizer.fit_transform(data['Title'])

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['To Predict: Intervention'], test_size=0.2, random_state=42)

# Step 5: Train a Random Forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = random_forest_classifier.predict(X_test)

# Step 7: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))

# Print a classification report for more details
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

num_unique_classes = data['To Predict: Intervention'].nunique()
print("Number of Unique Classes:", num_unique_classes)


# In[86]:


from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

# Calculate precision and recall
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print(classification_report(y_test, y_pred))


# In[97]:


#svm


# In[99]:


# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load and preprocess your dataset from an xlsx file
# Replace 'your_data.xlsx' with your dataset's filename
data = pd.read_excel('Data for ML.xlsx')

# Step 3: Vectorize the 'title' text data
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf_vectorizer.fit_transform(data['Title'])

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['To Predict: Study type'], test_size=0.2, random_state=42)

# Step 5: Train an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = svm_classifier.predict(X_test)

# Step 7: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Calculate precision, recall, and confusion matrix
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Print precision, recall, and confusion matrix
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=svm_classifier.classes_,
            yticklabels=svm_classifier.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

num_unique_classes = data['To Predict: Study type'].nunique()
print("Number of Unique Classes:", num_unique_classes)

# You can also print a classification report for more details
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[94]:


from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print(classification_report(y_test, y_pred))


# In[100]:


# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load and preprocess your dataset from an xlsx file
# Replace 'your_data.xlsx' with your dataset's filename
data = pd.read_excel('Data for ML.xlsx')

# Step 3: Vectorize the 'title' text data
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf_vectorizer.fit_transform(data['Title'])

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['To Predict: Disease Name'], test_size=0.2, random_state=42)

# Step 5: Train an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = svm_classifier.predict(X_test)

# Step 7: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Calculate precision, recall, and confusion matrix
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Print precision, recall, and confusion matrix
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=svm_classifier.classes_,
            yticklabels=svm_classifier.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

num_unique_classes = data['To Predict: Disease Name'].nunique()
print("Number of Unique Classes:", num_unique_classes)


# You can also print a classification report for more details
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[95]:


from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print(classification_report(y_test, y_pred))


# In[101]:


# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load and preprocess your dataset from an xlsx file
# Replace 'your_data.xlsx' with your dataset's filename
data = pd.read_excel('Data for ML.xlsx')

# Step 3: Vectorize the 'title' text data
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf_vectorizer.fit_transform(data['Title'])

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['To Predict: Intervention'], test_size=0.2, random_state=42)

# Step 5: Train an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = svm_classifier.predict(X_test)

# Step 7: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Calculate precision, recall, and confusion matrix
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Print precision, recall, and confusion matrix
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=svm_classifier.classes_,
            yticklabels=svm_classifier.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

num_unique_classes = data['To Predict: Intervention'].nunique()
print("Number of Unique Classes:", num_unique_classes)

# You can also print a classification report for more details
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[96]:


from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print(classification_report(y_test, y_pred))


# In[98]:


#abstarct


# In[110]:


# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score

# Step 2: Load and preprocess your dataset from an xlsx file
# Replace 'your_data.xlsx' with your dataset's filename
data = pd.read_excel('Data for ML.xlsx')
data['Abstract'].fillna('', inplace=True)

# Step 3: Print the number of unique classes in the 'study type' column
num_unique_classes = data['To Predict: Study type'].nunique()
print("Number of Unique Classes:", num_unique_classes)

# Step 4: Vectorize the 'title' text data
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf_vectorizer.fit_transform(data['Abstract'])

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['To Predict: Study type'], test_size=0.2, random_state=42)

# Step 6: Train a Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier(random_state=42)
decision_tree_classifier.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = decision_tree_classifier.predict(X_test)

# Step 8: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Calculate precision and recall
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# You can also print a classification report for more details
print(classification_report(y_test, y_pred))


# In[111]:


# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score

# Step 2: Load and preprocess your dataset from an xlsx file
# Replace 'your_data.xlsx' with your dataset's filename
data = pd.read_excel('Data for ML.xlsx')
data['Abstract'].fillna('', inplace=True)

# Step 3: Print the number of unique classes in the 'study type' column
num_unique_classes = data['To Predict: Study type'].nunique()
print("Number of Unique Classes:", num_unique_classes)

# Step 4: Vectorize the 'title' text data
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf_vectorizer.fit_transform(data['Abstract'])

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['To Predict: Disease Name'], test_size=0.2, random_state=42)

# Step 6: Train a Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier(random_state=42)
decision_tree_classifier.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = decision_tree_classifier.predict(X_test)

# Step 8: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Calculate precision and recall
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# You can also print a classification report for more details
print(classification_report(y_test, y_pred))


# In[79]:


# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score

# Step 2: Load and preprocess your dataset from an xlsx file
# Replace 'your_data.xlsx' with your dataset's filename
data = pd.read_excel('Data for ML.xlsx')
data['Abstract'].fillna('', inplace=True)

# Step 3: Print the number of unique classes in the 'study type' column
num_unique_classes = data['To Predict: Study type'].nunique()
print("Number of Unique Classes:", num_unique_classes)

# Step 4: Vectorize the 'title' text data
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf_vectorizer.fit_transform(data['Abstract'])

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['To Predict: Intervention'], test_size=0.2, random_state=42)

# Step 6: Train a Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier(random_state=42)
decision_tree_classifier.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = decision_tree_classifier.predict(X_test)

# Step 8: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Calculate precision and recall
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))

conf_matrix = confusion_matrix(y_test, y_pred)

dimensions = np.shape(conf_matrix)
rows, columns = dimensions


for i in range(rows):
    print(conf_matrix[i])

# You can also print a classification report for more details
print(classification_report(y_test, y_pred))


# In[21]:


#random forest


# In[102]:


# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Step 2: Load and preprocess your dataset from an xlsx file
# Replace 'your_data.xlsx' with your dataset's filename
data = pd.read_excel('Data for ML.xlsx')

# Drop rows with missing values in the 'title' column
data = data.dropna(subset=['Abstract'])

# Step 3: Vectorize the 'title' text data
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf_vectorizer.fit_transform(data['Abstract'])

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['To Predict: Study type'], test_size=0.2, random_state=42)

# Step 5: Train a Random Forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = random_forest_classifier.predict(X_test)

# Step 7: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# You can also print a classification report for more details
print(classification_report(y_test, y_pred))

num_unique_classes = data['To Predict: Study type'].nunique()
print("Number of Unique Classes:", num_unique_classes)

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)


# In[103]:


# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
import numpy as np

# Step 2: Load and preprocess your dataset from an xlsx file
# Replace 'your_data.xlsx' with your dataset's filename
data = pd.read_excel('Data for ML.xlsx')

# Drop rows with missing values in the 'title' column
data = data.dropna(subset=['Abstract'])

# Step 3: Vectorize the 'title' text data
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf_vectorizer.fit_transform(data['Abstract'])

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['To Predict: Disease Name'], test_size=0.2, random_state=42)

# Step 5: Train a Random Forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = random_forest_classifier.predict(X_test)


# Step 7: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))

# You can also print a classification report for more details
print(classification_report(y_test, y_pred))

num_unique_classes = data['To Predict: Disease Name'].nunique()
print("Number of Unique Classes:", num_unique_classes)

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)


# In[77]:


# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
import numpy as np

# Step 2: Load and preprocess your dataset from an xlsx file
# Replace 'your_data.xlsx' with your dataset's filename
data = pd.read_excel('Data for ML.xlsx')

# Drop rows with missing values in the 'title' column
data = data.dropna(subset=['Abstract'])

# Step 3: Vectorize the 'title' text data
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf_vectorizer.fit_transform(data['Abstract'])

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['To Predict: Intervention'], test_size=0.2, random_state=42)

# Step 5: Train a Random Forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = random_forest_classifier.predict(X_test)


# Step 7: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))

# You can also print a classification report for more details
print(classification_report(y_test, y_pred))

num_unique_classes = data['To Predict: Intervention'].nunique()
print("Number of Unique Classes:", num_unique_classes)

conf_matrix = confusion_matrix(y_test, y_pred)

dimensions = np.shape(conf_matrix)
rows, columns = dimensions


for i in range(rows):
    print(conf_matrix[i])


# In[11]:


#svm


# In[105]:


# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

# Step 2: Load and preprocess your dataset from an xlsx file
# Replace 'your_data.xlsx' with your dataset's filename
data = pd.read_excel('Data for ML.xlsx')

# Step 3: Handle missing values (NaN) in 'title' column
data['Abstract'].fillna('', inplace=True)

# Step 4: Vectorize the 'title' text data
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf_vectorizer.fit_transform(data['Abstract'])

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['To Predict: Study type'], test_size=0.2, random_state=42)

# Step 6: Train an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = svm_classifier.predict(X_test)

# Step 8: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Calculate precision and recall
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))

num_unique_classes = data['To Predict: Study type'].nunique()
print("Number of Unique Classes:", num_unique_classes)

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# You can also print a classification report for more details
print(classification_report(y_test, y_pred))


# In[106]:


# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

# Step 2: Load and preprocess your dataset from an xlsx file
# Replace 'your_data.xlsx' with your dataset's filename
data = pd.read_excel('Data for ML.xlsx')

# Step 3: Handle missing values (NaN) in 'title' column
data['Abstract'].fillna('', inplace=True)

# Step 4: Vectorize the 'title' text data
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf_vectorizer.fit_transform(data['Abstract'])

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['To Predict: Disease Name'], test_size=0.2, random_state=42)

# Step 6: Train an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = svm_classifier.predict(X_test)

# Step 8: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Calculate precision and recall
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))

num_unique_classes = data['To Predict: Disease Name'].nunique()
print("Number of Unique Classes:", num_unique_classes)

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# You can also print a classification report for more details
print(classification_report(y_test, y_pred))


# In[78]:


# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

# Step 2: Load and preprocess your dataset from an xlsx file
# Replace 'your_data.xlsx' with your dataset's filename
data = pd.read_excel('Data for ML.xlsx')

# Step 3: Handle missing values (NaN) in 'title' column
data['Abstract'].fillna('', inplace=True)

# Step 4: Vectorize the 'title' text data
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf_vectorizer.fit_transform(data['Abstract'])

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['To Predict: Intervention'], test_size=0.2, random_state=42)

# Step 6: Train an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = svm_classifier.predict(X_test)

# Step 8: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Calculate precision and recall
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))

num_unique_classes = data['To Predict: Intervention'].nunique()
print("Number of Unique Classes:", num_unique_classes)

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

dimensions = np.shape(conf_matrix)
rows, columns = dimensions


for i in range(rows):
    print(conf_matrix[i])

# You can also print a classification report for more details
print(classification_report(y_test, y_pred))


# In[ ]:




