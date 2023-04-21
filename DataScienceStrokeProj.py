#!/usr/bin/env python
# coding: utf-8

# In[277]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import sklearn
import warnings
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, f1_score
from sklearn.compose import ColumnTransformer
warnings.filterwarnings('ignore')
from sklearn import tree
import pydotplus 
from sklearn.tree import export_graphviz
import subprocess
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn import metrics
from seaborn import heatmap


# In[278]:


#הוצאת הדאטה לתוך משתנה לצורך עבודה


# In[279]:


df = pd.read_csv('healthcare-dataset-stroke-data.csv', index_col=0)
df


# In[280]:


df.head()


# In[281]:


df.dtypes


# In[282]:


# בדיקת חוסרים: 
#   ניתן לראות שבעמודה מספר 8  ישנם 201 עמודות בהם ישנו חוסר במידע 


# In[283]:


df.info()


# In[284]:


#   אנו נצטרך לטפל בחוסרים ע"י מחיקה או הוספת שורות נתונים חסרים , אני בחרתי למחוק את השורות בהם חסרים נתונים   


# In[285]:


#   ניתן לראות את כמות הנבדקים וטווח של כל פרמטר בטבלה:


# In[286]:


df.describe(include='all')


# In[287]:


df = df.dropna()


# In[288]:


#   בדיקה שאכן המחיקה התצבעה בהצלחה


# In[289]:


df.info()


# In[290]:


#   בדיקה שאכן המחיקה התצבעה בהצלחעל פי דיאגרמת עוגה ניתן לראות שרוב העובדים הינם שכירים ,מתוכם בערך 16 אחוזים עצמאיים


# In[291]:


work_type = df['work_type'].value_counts()
fig = px.pie(values=work_type.values, names=work_type.index, title='פיזור סוג עבודה',color_discrete_sequence=px.colors.qualitative.Pastel1)
fig.show()


# In[292]:


#  נבצע המרה של העמודות הקטגוריליות למספרים טבעיים מ0 עד כמות הקטגורויות באותו המשתנה לצורך עבודה,הצגה וניתוח נוח וברור 


# In[293]:


lab = LabelEncoder()
label_df = pd.DataFrame(columns=df.columns)

for col in df.columns:
    label_df[col] = lab.fit_transform(df[col])
    


# In[ ]:


# בדיקה שעבר המרת המשתנים בצורה טובה 
#
#
#
#
#
#
#


# In[294]:


label_df.info()


# In[295]:


label_df.head()


# In[ ]:


#שכיחות 


# In[296]:


pn =df.select_dtypes(include=[np.number])
pn.hist(figsize=(10,10), layout=(7,3), xlabelsize=10, ylabelsize=10);


# In[297]:


plt.title('Gender Distribution')
df.gender.value_counts().plot(kind='pie',autopct='%1.1f%%')


# In[298]:


plt.title('Ever Married Distribution')
df.ever_married.value_counts().plot(kind='pie',autopct='%1.1f%%')


# In[299]:


plt.title('Work Type Distribution')
df.work_type.value_counts().plot(kind='pie',autopct='%1.1f%%')


# In[300]:


plt.title('Residence Type Distribution')
df.Residence_type.value_counts().plot(kind='pie',autopct='%1.1f%%')


# In[301]:


plt.title('Smoking Status Distribution')
df.smoking_status.value_counts().plot(kind='pie',autopct='%1.1f%%')


# In[302]:


plt.title('Heart Disease Distribution')
df.heart_disease.value_counts().plot(kind='pie',autopct='%1.1f%%')


# In[303]:


plt.title('Hypertensione Distribution')
df.hypertension.value_counts().plot(kind='pie',autopct='%1.1f%%')


# In[ ]:


#   המרנו את המשתנים הקטוגוריאלים למשתנים נומריים לצורך עבודה יותר קלה ברגרסיה לוגיסטית 
#
#
#
#
#
#
#


# In[304]:


categorical = ['gender', 'hypertension', 'heart_disease', 'ever_married',
               'work_type', 'Residence_type', 'smoking_status']
numerical = ['age','avg_glucose_level', 'bmi', 'stroke']
# Define the transformers for numerical and categorical features
num_transformer = ColumnTransformer(transformers=[('num', 'passthrough', numerical)])
cat_transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical)])
# Fit and transform the transformers on the data
num_features = num_transformer.fit_transform(df)
cat_features = cat_transformer.fit_transform(df)
# Concatenate the transformed features into a single DataFrame
new_df = pd.DataFrame(data=np.hstack([num_features, cat_features]))
# Rename columns with new column names
new_columns = numerical + cat_transformer.named_transformers_['cat'].get_feature_names(categorical).tolist()
new_df.columns = new_columns
# Display the concatenated DataFrame
new_df.head()


# In[305]:


sns.heatmap(df.corr(), annot=True)


# In[306]:


#  מפת חום לזיהוי קורלאציות חיובית, ניתן לראות שגיל הוא פקטור משמעותי  


# In[307]:


fig=plt.figure()
ax=plt.axes()
ax.scatter(df.bmi,df.age)
plt.title('Age And BMI On The Axis,Red=Had A Stroke ')
plt.xlabel('BMI')
plt.ylabel('Age')
ax.scatter(df.bmi[df.stroke==1],df.age[df.stroke==1],c='red')
plt.show()


# In[308]:


df1=df.copy()
Age1= df1.loc[df1.stroke == 1,'age']
Age1.describe()


# In[309]:



# Create a figure and axis object
fig, ax = plt.subplots(figsize=(12, 8))
# Plot the box plot
sns.boxplot(x='stroke', y='age', data=df1, ax=ax)
# Set the title
ax.set_title('Age')
# Show the plot
plt.show()


# In[310]:


# נרמול הנתונים בעזרת StandardScaler
#טווחי הערכים בכל משתנה הם שונים. נרצה לנרמל את הנתונים ולהתאים אותם לטווח בין 1- ל1 כדי שנוכל לחזות ולסווג את הנתונים טוב יותר בעזרת המודלים 


# In[311]:


df2 = new_df.copy()
for col in df2.drop(['stroke'], axis=1).columns:
  scaler = StandardScaler()
  df2[col] = scaler.fit_transform(df2[col].values.reshape(-1, 1))
    


# In[312]:


df2.head()


# In[313]:


X = df2.drop(columns=['stroke'],axis=1)
y = df2['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)


# In[314]:


model = LogisticRegression()
model.fit(X_train,y_train)
prediction3=model.predict(X_test)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,y_test))


# In[315]:


print(classification_report(y_test, prediction3))


# In[316]:


def splitData(features):
    """Split a subset of the titanic dataset, given by the features, into train and test sets."""
    df_predictors = df[features].values
    df_labels = df["stroke"].values

    # Split into training and test sets
    XTrain, XTest, yTrain, yTest = train_test_split(df_predictors, df_labels, random_state=1, test_size=0.3)
    return XTrain, XTest, yTrain, yTest


# In[317]:


def renderTree(my_tree, features):
    # hacky solution of writing to files and reading again
    # necessary due to library bugs
    filename = "temp.dot"
    with open(filename, 'w') as f:
        f = tree.export_graphviz(my_tree, 
                                 out_file=f, 
                                 feature_names=features, 
                                 class_names=["stroke", "NOT stroke"],  
                                 filled=True, 
                                 rounded=True,
                                 special_characters=True)
  
    dot_data = ""
    with open(filename, 'r') as f:
        dot_data = f.read()

    graph = pydotplus.graph_from_dot_data(dot_data)
    image_name = "temp.png"
    graph.write_png(image_name)  
    display(Image(filename=image_name))


# In[318]:


decisionTree = tree.DecisionTreeClassifier()

XTrain, XTest, yTrain, yTest = splitData(["age"])
# fit the tree with the traing data
decisionTree = decisionTree.fit(XTrain, yTrain)

# predict with the training data
y_pred_train = decisionTree.predict(XTrain)
# measure accuracy
print('Accuracy on training data = ', 
      metrics.accuracy_score(y_true = yTrain, y_pred = y_pred_train))

# predict with the test data
y_pred = decisionTree.predict(XTest)
# measure accuracy
print('Accuracy on test data = ', 
      metrics.accuracy_score(y_true = yTest, y_pred = y_pred))

renderTree(decisionTree, ["age"])


# In[319]:


df = pd.read_csv('healthcare-dataset-stroke-data.csv')
x = df2.drop(columns=['stroke'],axis=1)
y = df2['stroke']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,random_state = 1)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5,random_state = 1) 


# In[320]:


neigh = KNeighborsClassifier(n_neighbors=25)
neigh.fit(x_train, y_train)
prediction=neigh.predict(x_val)
fpr, tpr, _ = metrics.roc_curve(y_val,prediction)
y_scores = neigh.predict_proba(x_val)
fpr, tpr, threshold = roc_curve(y_val, y_scores[:, 1])
rocAuc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'c', label = 'AUC = %0.2f' % rocAuc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Graph  ')
plt.show()


# In[325]:


auc = metrics.roc_auc_score(y_val, prediction)
print(classification_report(y_val, prediction))
matrix = confusion_matrix(y_val,prediction)
heatmap(matrix,annot=True, fmt="f")


# In[322]:


Logistic = LogisticRegression()
Logistic.fit(x_train,y_train)
prediction=Logistic.predict(x_val)
fpr, tpr, _ = metrics.roc_curve(y_val,prediction)
y_scores = Logistic.predict_proba(x_val)
fpr, tpr, threshold = roc_curve(y_val, y_scores[:, 1])
rocAuc = auc(fpr, tpr)


# In[323]:


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'c', label = 'AUC = %0.2f' % rocAuc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Graph')
plt.show()


# In[324]:


auc = metrics.roc_auc_score(y_val, prediction)
print(classification_report(y_val, prediction))
matrix = confusion_matrix(y_val,prediction)
heatmap(matrix,annot=True, fmt="f")


# In[ ]:




