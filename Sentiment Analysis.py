#!/usr/bin/env python
# coding: utf-8

# # Text Classification Project
# Now we're at the point where we should be able to:
# * Read in a collection of documents - a *corpus*
# * Transform text into numerical vector data using a pipeline
# * Create a classifier
# * Fit/train the classifier
# * Test the classifier on new data
# * Evaluate performance

# ## Perform imports and load the dataset
# The dataset contains the text of 2000 movie reviews. 1000 are positive, 1000 are negative, and the text has been preprocessed as a tab-delimited file.

# In[1]:


import pandas as pd
import numpy as np


# In[35]:


df=pd.read_csv(r"C:\Users\Divyashree K\Downloads\UPDATED_NLP_COURSE\UPDATED_NLP_COURSE\TextFiles\moviereviews.tsv",sep='\t')


# In[3]:


df.head()


# In[4]:


len(df)


# In[5]:


df['label'].value_counts()


# ## To check whether data  is balanced or unbalanced
# 
# Unbalanced data brings challenges to feature correlation, class separation and evaluation, and results in poor model performance.
# 
# 

# In[6]:


df['label'].value_counts()      # balanced data set


# ## Check for missing values:
# The data might have Some have NaN values, others have short strings composed of only spaces. This might happen if a reviewer declined to provide a comment with their review. We will show two ways using pandas to identify and remove records containing empty data.
# * NaN records are efficiently handled with [.isnull()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.isnull.html) and [.dropna()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html)
# * Strings that contain only whitespace can be handled with [.isspace()](https://docs.python.org/3/library/stdtypes.html#str.isspace), [.itertuples()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.itertuples.html), and [.drop()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html)
# 
# ### Detect & remove NaN values:

# In[7]:


# Check for the existence of NaN values in a cell:
df.isnull().sum()


# 35 reviews are containing **NaN** value, these are removed by using `.dropna()` pandas function.
# 
# <div class="alert-info" style="margin: 15px">CAUTION: By setting inplace=True, we permanently affect the DataFrame currently in memory, and this can't be undone. However, it does *not* affect the original source data. If we needed to, we could always load the original DataFrame from scratch.</div>

# In[8]:


df.dropna(inplace=True)
len(df)


# In[9]:


print(df.isnull().sum())


# ### Detect & remove empty strings
# Technically, we're dealing with "whitespace only" strings. If the original .tsv file had contained empty strings, pandas **.read_csv()** would have assigned NaN values to those cells by default.
# 
# In order to detect these strings we need to iterate over each row in the DataFrame. The **.itertuples()** pandas method is a good tool for this as it provides access to every field. For brevity we'll assign the names `i`, `lb` and `rv` to the `index`, `label` and `review` columns.

# In[10]:


blanks=[]    # start with an empty list

for i,ib,rv in df.itertuples():  # iterate over the DataFrame
    if type(rv)==str:            # avoid NaN values
        if rv.isspace():         # test 'review' for whitespace
            blanks.append(i)     # # add matching index numbers to the list
            
print(len(blanks))
print('blanks',blanks)


# Next we'll pass our list of index numbers to the **.drop()** method, and set `inplace=True` to make the change permanent.

# In[11]:


df.drop(blanks,inplace=True)


# In[12]:


len(df)


# Great! We dropped 62 records from the original 2000. Let's continue with the analysis.

# ## Take a quick look at the columns

# In[13]:


df['label'].value_counts()


# ## split the data into train and test

# In[20]:


from sklearn.model_selection import train_test_split

X = df['review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ## Build pipelines to vectorize the data, then train and fit a model
# Now that we have sets to train and test, we'll develop a selection of pipelines, each with a different model.

# In[21]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Naïve Bayes:
text_clf_nb = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', MultinomialNB()),
])

# Linear SVC:

text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),
])


# ## Feed the training data through the first pipeline

# In[22]:


text_clf_nb.fit(X_train, y_train)


# ## Run predictions and analyze the results (naïve Bayes)

# In[23]:


# Form a prediction set
predictions = text_clf_nb.predict(X_test)


# In[24]:


# Report the confusion matrix
from sklearn import metrics
print(metrics.confusion_matrix(y_test,predictions))


# In[25]:


# Print a classification report
print(metrics.classification_report(y_test,predictions))


# In[26]:


# Print the overall accuracy
print(metrics.accuracy_score(y_test,predictions))


# Naïve Bayes gave us better-than-average results at 76.4% for classifying reviews as positive or negative based on text alone. Let's see if we can do better.

# ## Feed the training data through the second pipeline
# Next we'll run Linear SVC

# In[27]:


text_clf_lsvc.fit(X_train, y_train)


# ## Run predictions and analyze the results (Linear SVC)

# In[28]:


# Form a prediction set
predictions = text_clf_lsvc.predict(X_test)


# In[29]:


# Report the confusion matrix
from sklearn import metrics
print(metrics.confusion_matrix(y_test,predictions))


# In[30]:


# Print a classification report
print(metrics.classification_report(y_test,predictions))


# In[31]:


# Print the overall accuracy
print(metrics.accuracy_score(y_test,predictions))


# Not bad! Based on text alone we correctly classified reviews as positive or negative **84.7%** of the time. In an upcoming section we'll try to improve this score even further by performing *sentiment analysis* on the reviews.

# ## Advanced Topic - Adding Stopwords to CountVectorizer
# By default, **CountVectorizer** and **TfidfVectorizer** do *not* filter stopwords. However, they offer some optional settings, including passing in your own stopword list.
# <div class="alert alert-info" style="margin: 20px">CAUTION: There are some [known issues](http://aclweb.org/anthology/W18-2502) using Scikit-learn's built-in stopwords list. Some words that are filtered may in fact aid in classification. In this section we'll pass in our own stopword list, so that we know exactly what's being filtered.</div>

# The [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) class accepts the following arguments:
# > *CountVectorizer(input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, **stop_words=None**, token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=<class 'numpy.int64'>)*
# 
# [TfidVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) supports the same arguments and more. Under *stop_words* we have the following options:
# > stop_words : *string {'english'}, list, or None (default)*
# 
# That is, we can run `TfidVectorizer(stop_words='english')` to accept scikit-learn's built-in list,<br>
# or `TfidVectorizer(stop_words=[a, and, the])` to filter these three words. In practice we would assign our list to a variable and pass that in instead.

# Scikit-learn's built-in list contains 318 stopwords:
# > <pre>from sklearn.feature_extraction import text
# > print(text.ENGLISH_STOP_WORDS)</pre>
# ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves']
# 
# However, there are words in this list that may influence a classification of movie reviews. With this in mind, let's trim the list to just 60 words:

# In[34]:


stopwords = ['a', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'can', \
             'even', 'ever', 'for', 'from', 'get', 'had', 'has', 'have', 'he', 'her', 'hers', 'his', \
             'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'me', 'my', 'of', 'on', 'or', \
             'see', 'seen', 'she', 'so', 'than', 'that', 'the', 'their', 'there', 'they', 'this', \
             'to', 'was', 'we', 'were', 'what', 'when', 'which', 'who', 'will', 'with', 'you']


# Now let's repeat the process above and see if the removal of stopwords improves or impairs our score.

# In[36]:


# YOU DO NOT NEED TO RUN THIS CELL UNLESS YOU HAVE
# RECENTLY OPENED THIS NOTEBOOK OR RESTARTED THE KERNEL:

import numpy as np
import pandas as pd

df=pd.read_csv(r"C:\Users\Divyashree K\Downloads\UPDATED_NLP_COURSE\UPDATED_NLP_COURSE\TextFiles\moviereviews.tsv",sep='\t')
df.dropna(inplace=True)
blanks = []
for i,lb,rv in df.itertuples():
    if type(rv)==str:
        if rv.isspace():
            blanks.append(i)
df.drop(blanks, inplace=True)
from sklearn.model_selection import train_test_split
X = df['review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics


# In[37]:


# RUN THIS CELL TO ADD STOPWORDS TO THE LINEAR SVC PIPELINE:
text_clf_lsvc2 = Pipeline([('tfidf', TfidfVectorizer(stop_words=stopwords)),
                     ('clf', LinearSVC()),
])
text_clf_lsvc2.fit(X_train, y_train)


# In[38]:


predictions = text_clf_lsvc2.predict(X_test)
print(metrics.confusion_matrix(y_test,predictions))


# In[39]:


print(metrics.classification_report(y_test,predictions))


# In[40]:


print(metrics.accuracy_score(y_test,predictions))


# Our score didn't change that much. We went from 84.7% without filtering stopwords to 84.4% after adding a stopword filter to our pipeline. Keep in mind that 2000 movie reviews is a relatively small dataset. The real gain from stripping stopwords is improved processing speed; depending on the size of the corpus, it might save hours.

# ## Feed new data into a trained model
# Once we've developed a fairly accurate model, it's time to feed new data through it. In this last section we'll write our own review, and see how accurately our model assigns a "positive" or "negative" label to it.

# In[41]:


# Naïve Bayes Model:
text_clf_nb = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', MultinomialNB()),
])

# Linear SVC Model:
text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),
])

# Train both models on the moviereviews.tsv training set:
text_clf_nb.fit(X_train, y_train)
text_clf_lsvc.fit(X_train, y_train)


# ### Next, feed new data to the model's `predict()` method

# In[42]:


myreview = "A movie I really wanted to love was terrible. \
I'm sure the producers had the best intentions, but the execution was lacking."


# In[43]:


print(text_clf_nb.predict([myreview]))  # be sure to put "myreview" inside square brackets


# In[44]:


print(text_clf_lsvc.predict([myreview]))


# Great! Now you should be able to build text classification pipelines in scikit-learn

# In[ ]:




