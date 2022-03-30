#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import joblib
from flask import Flask


# In[17]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv('Iris.csv')
print(df.head())


# In[3]:


X = df.loc[:, ~df.columns.isin(['Id' , 'Species'])]
print(X.head())


# In[4]:


y = df['Species']
print(y.head())


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

print("Training Dataset Shape: ", X_train.shape)
print(X_train.head())

print("Testing Dataset Shape: ", X_test.shape)
print(X_test.head())

print("Training Dataset Shape: ", y_train.shape)
print(y_train.head())

print("Testing Dataset Shape: ", y_test.shape)
print(y_test.head())


# In[7]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()


# In[8]:


model.fit(X_train, y_train)

model.score(X_train, y_train)


# In[9]:


predict = model.predict(X_test)

predict[:10]


# In[10]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print("Accuracy Score:")
print(accuracy_score(y_test, predict))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predict))

print("Classification Report:")
print(classification_report(y_test, predict))


# In[11]:


import joblib
joblib.dump(model, './randomforest_model.pkl')


# In[13]:


test_data = [5.1, 3.2, 1.5, 0.4]

test_data = np.array(test_data)

test_data = test_data.reshape(1, -1)
print(test_data)


# In[14]:


file = open('randomforest_model.pkl' , "rb" )


# In[15]:


trained_model = joblib.load(file)


# In[18]:


pred = trained_model.predict(test_data)
print(pred)


# In[19]:


from os import mkdir
get_ipython().system('mkdir flask-app')


# In[20]:


from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
  return "Hello World"

if __name__ == '__main__':
  app.run(debug=True)


# In[21]:



#import Flask
from flask import Flask, render_template
#create an instance of Flask
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')
if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




