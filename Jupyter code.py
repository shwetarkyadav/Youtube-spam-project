#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# ML Packages For Vectorization of Text For Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[5]:


# Dataset from https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection#
df1 = pd.read_csv("C:/Users/yoges/Desktop/Youtube01-Psy.csv")


# In[6]:


df1.head()


# In[7]:


# Load all our dataset to merge them
df2 = pd.read_csv("C:/Users/yoges/Desktop/Youtube02-KatyPerry.csv")
df3 = pd.read_csv("C:/Users/yoges/Desktop/Youtube03-LMFAO.csv")
df4 = pd.read_csv("C:/Users/yoges/Desktop/Youtube04-Eminem.csv")
df5 = pd.read_csv("C:/Users/yoges/Desktop/Youtube05-Shakira.csv")


# In[8]:


frames = [df1,df2,df3,df4,df5]


# In[9]:


# Merging or Concatenating our DF
df_merged = pd.concat(frames)


# In[10]:


df_merged


# In[11]:


# Total Size
df_merged.shape


# In[12]:


# Merging with Keys
keys = ["Psy","KatyPerry","LMFAO","Eminem","Shakira"]


# In[13]:


df_with_keys = pd.concat(frames,keys=keys)


# In[14]:


df_with_keys


# In[15]:


# Checking for Only Comments on Shakira
df_with_keys.loc['Shakira']


# In[16]:


# Save and Write Merged Data to csv
df_with_keys.to_csv("YoutubeSpamMergeddata.csv")


# In[17]:


df = df_with_keys


# In[18]:


df.size


# In[19]:


# Checking for Consistent Column Name
df.columns


# In[20]:


# Checking for Datatypes
df.dtypes


# In[21]:


# Check for missing nan
df.isnull().isnull().sum()


# In[22]:


# Checking for Date
df["DATE"]


# In[23]:


df.AUTHOR
# Convert the Author Name to First Name and Last Name
#df[["FIRSTNAME","LASTNAME"]] = df['AUTHOR'].str.split(expand=True)


# In[24]:


df_data = df[["CONTENT","CLASS"]]


# In[25]:


df_data.columns


# In[26]:


df_x = df_data['CONTENT']
df_y = df_data['CLASS']


# In[27]:


cv = CountVectorizer()
ex = cv.fit_transform(["Great song but check this out","What is this song?"])


# In[28]:


ex.toarray()


# In[29]:


cv.get_feature_names()


# In[30]:


# Extract Feature With CountVectorizer
corpus = df_x
cv = CountVectorizer()
X = cv.fit_transform(corpus) # Fit the Data


# In[31]:


X.toarray()


# In[32]:


# get the feature names
cv.get_feature_names()


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)


# In[35]:


X_train


# In[36]:


# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


# In[37]:


# Accuracy of our Model
print("Accuracy of Model",clf.score(X_test,y_test)*100,"%")


# In[38]:


## Predicting with our model
clf.predict(X_test)


# In[39]:


# Sample Prediciton
comment = ["Check this out"]
vect = cv.transform(comment).toarray()


# In[40]:


clf.predict(vect)


# In[41]:


class_dict = {'ham':0,'spam':1}


# In[42]:


class_dict.values()


# In[43]:


if clf.predict(vect) == 1:
    print("Spam")
else:
    print("Ham")


# In[44]:


# Sample Prediciton 2
comment1 = ["Great song Friend"]
vect = cv.transform(comment1).toarray()
clf.predict(vect)


# In[45]:


import pickle


# In[46]:


naivebayesML = open("YtbSpam_model.pkl","wb")


# In[47]:


pickle.dump(clf,naivebayesML)


# In[48]:


naivebayesML.close()


# In[49]:


ytb_model = open("YtbSpam_model.pkl","rb")


# In[50]:


new_model = pickle.load(ytb_model)


# In[51]:


new_model


# In[56]:


# Sample Prediciton 3
comment2 = ["great song"]
vect = cv.transform(comment2).toarray()
new_model.predict(vect)


# In[57]:


if new_model.predict(vect) == 1:
    print("Spam")
else:
    print("Ham")


# In[ ]:




