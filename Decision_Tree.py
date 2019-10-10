#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

# In[2]:


filename = 'D:\\Work_Space\\Machine_Learning\\Decision_Tree\\play_data.csv'


# In[3]:
def train_model(filename):
    filename = filename.replace('/', '\\')
    print(filename)
    play = pd.read_csv(filename)

    # In[22]:

    play

    # In[5]:

    name_label = play.columns[len(play.columns) - 1]

    # In[6]:

    # Categorial boolean mask
    categorial_feature_mask = play.dtypes == object
    # Filter categorial columns using mask and turn into a list
    categorial_cols = play.columns[categorial_feature_mask].tolist()

    # In[7]:

    X_play = pd.get_dummies(play.iloc[:, 0:len(play.columns)-1],
                            columns=categorial_cols[:len(play.columns)-1])

    # In[8]:

    X_play

    # In[9]:

    # Label Encoder
    label_encoder = LabelEncoder()

    # Inverse yes-no into 1-0
    y_play = label_encoder.fit_transform(play[name_label])

    # In[10]:

    # Instantiate DecisionTreeClassifier
    dTreeClf = DecisionTreeClassifier(criterion='entropy', max_depth=3)

    # In[11]:

    X_train, X_test, y_train, y_test = train_test_split(
        X_play, y_play, test_size=0.2, random_state=2)
    # Train model using DecisionTreeClassifier
    dTreeClf.fit(X_train, y_train)

    # In[12]:
    predictions = dTreeClf.predict(X_test)
    # creating a confusion matrix
    cm = confusion_matrix(y_test, predictions)
    count = (cm[0][0]+cm[1][1])/(cm[0][1]+cm[1][0] + cm[0][0] + cm[1][1])*100
    print("-"*30)
    print('Currency is %.2f' % count)

    dot_data = StringIO()
    export_graphviz(dTreeClf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True,
                    feature_names=X_play.columns,
                    class_names=label_encoder.classes_)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('D:\\Work_Space\\Machine_Learning\\Decision_Tree\\diabetes.png')
    Image(graph.create_png())
    play_label = play.iloc[:, 0:len(play.columns)-1].columns
    plays = play.iloc[:, 0:len(play.columns)-1]
    return count, plays, X_play.columns, label_encoder, dTreeClf


#columns, train_label, label_encoder, dTreeClf = train_model(filename)
'''=================================================================================='''

# test_play = pd.DataFrame([['sunny', 'hot', 'high', 'weak']])
# test_play.columns = columns
#
# # Get_dummies
# test_play = pd.get_dummies(test_play, columns=test_play.columns)
#
# missing_column = set(train_label) - set(test_play.columns)
# for c in missing_column:
#     test_play[c] = 0
#
# X_test = test_play[train_label]
# print(X_test.values)
#
# predictions = dTreeClf.predict(X_test)
#
#
# # In[19]:
#
#
# predictions = label_encoder.inverse_transform(predictions)
#
#
# # In[20]:
#
#
# print(predictions)


# Job : type of job (categorical: 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown')
# Marital : marital status (categorical: 'divorced', 'married', 'single', 'unknown' ; note: 'divorced' means divorced or widowed)
# Education (categorical: 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown')
# Default: has credit in default? (categorical: 'no', 'yes', 'unknown')
# Housing: has housing loan? (categorical: 'no', 'yes', 'unknown')
# Loan: has personal loan? (categorical: 'no', 'yes', 'unknown')
