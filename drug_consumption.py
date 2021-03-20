from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import truncnorm, randint
from scipy import stats
from modelling_functions import *
from preprocessing_functions import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
shap.initjs()
%matplotlib inline

## Data Preprocessing
from sklearn.preprocessing import LabelEncoder

# Load input data
col_names = ['ID','Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 
             'Cscore', 'Impulsive', 'SS', 'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 
             'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh','LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA'] 
df = pd.read_csv('drug_consumption.data', header = None, names = col_names)

# Remove uneeded columns
to_drop = ["ID","Semer"]
df = df.drop(to_drop, axis = 1)

# Convert categorical columns into numeric
le = LabelEncoder()
for col in df:
    if df[col].dtype == 'object':
        df[col]=le.fit_transform(df[col])

# Convert multi-class (Class 0 to Class 6) into binary
df.replace(to_replace =[0, 1, 2], value = 0, inplace = True) #Non Users
df.replace(to_replace =[3, 4, 5, 6], value = 1, inplace = True)  #Users

# Separate into feature set and drug set
df_features = df.drop(df.loc[:, 'Alcohol':'VSA'].columns, axis = 1) 
df_drugs = df.drop(df.loc[:, 'Age':'SS'].columns, axis = 1) 

## Analysis for Drug type: Alcohol. The process was repeated for other types of drugs as well.

# create subset
Alc = subset(0, "Alcohol")
# split data into train and test set. Apply smote to handle class imbalance in the train set only.
X_train0, y_train0, X_test0, y_test0 = split_smote(Alc, "Alcohol")
# model training data using four algorithms
lg_0, tree_0, rf_0, svm_0 = model_all(X_train0, y_train0, X_test0)
# identify best performing algorithm and evaluate using test set
y_pred0 = svm_0.predict(X_test0)
# gain further evaluation using confusion matrix
cm_0 = confusion_matrix(y_test0, y_pred0)
results_cm(cm_0, X_test0, y_test0, svm_0)

# Use shap to gain model centric explanations
explainer_0 = shap.KernelExplainer(svm_0.predict, X_train0)
shap_values0 = explainer_0.shap_values(X_test0)
# Use explainer object to gain different insights into feature behavior 
#Summary Plot as a bar chart
shap.summary_plot(shap_values = shap_values0, features = X_test0, max_display=12, plot_type='bar')
shap.summary_plot(shap_values0, X_test0)
#Dependence plot to showcase relationships w.r.t. a particular feature
shap.dependence_plot("Nscore", shap_values0, X_test0)
# plot the SHAP values for the 10th observation to observe how feature behavior varies
shap.force_plot(explainer_0.expected_value,shap_values0[10,:], X_test0.iloc[5,:])

# Use shap as feature selection mechanism
# Idenitfy most influencial features in the explanations and train a new model to observe results.
new_model0 = model_svm(train_X0, y_train0, test_X0)
y_pred_new0 = new_model0.predict(test_X0)
cm_new0 = confusion_matrix(y_test0, y_pred_new0)
results_cm(cm_new0, test_X0 , y_test0, new_model0)