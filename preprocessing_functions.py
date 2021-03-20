from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


#data preprocessing functions
def subset(drug_no,drug_name):
    part = df_drugs.iloc[:,drug_no]
    df2 = df_features.assign(drug = part)
    df2.rename(columns = {'drug':drug_name}, inplace = True)
    return df2

def split(drug_df, drug_name):
    X = drug_df.drop([drug_name], axis = 1)
    y = drug_df[drug_name]
    counter = Counter(y)
    print('The distribution of classes is: {}'.format(counter))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42, stratify = y)
    return X_train, y_train, X_test, y_test


def split_smote_2(drug_df, drug_name):
    X = drug_df.drop([drug_name], axis = 1)
    y = drug_df[drug_name]
    counter = Counter(y)
    print('Originally, the distribution of classes is: {}'.format(counter))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42, stratify = y)
    over = SMOTE(sampling_strategy=0.8)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('u', under), ('o',over)]
    pipeline = Pipeline(steps=steps)
    Xsm_train, ysm_train = pipeline.fit_resample(X_train, y_train)
    counter_balance = Counter(ysm_train)
    print('After SMOTE sampling, the distribution of classes in Training set is: {}'.format(counter_balance))
    XSM_train = pd.DataFrame(Xsm_train, columns=X_train.columns)
    return XSM_train, ysm_train, X_test, y_test

#For the highly imbalanced drugs 
def split_smote(drug_df, drug_name):
    X = drug_df.drop([drug_name], axis = 1)
    y = drug_df[drug_name]
    counter = Counter(y)
    print('Originally, the distribution of classes is: {}'.format(counter))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42, stratify = y)
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    Xsm_train, ysm_train = pipeline.fit_resample(X_train, y_train)
    counter_balance = Counter(ysm_train)
    print('After SMOTE sampling, the distribution of classes in Training set is: {}'.format(counter_balance))
    XSM_train = pd.DataFrame(Xsm_train, columns=X_train.columns)
    return XSM_train, ysm_train, X_test, y_test

def scale(X_train, X_test):
    scaler = StandardScaler()
    rescaledX_train = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(rescaledX_train, columns=X_train.columns)
    rescaledX_test = scaler.fit_transform(X_test)
    X_test_scaled = pd.DataFrame(rescaledX_test, columns=X_test.columns)
    return X_train_scaled, X_test_scaled

def results_cm(cm, X_test, y_test,model):
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    annot_kws = {"ha": 'center',"va": 'bottom', "color":'black',"size": 15}
    sns.heatmap(cm, annot=labels, fmt='' , cmap='Blues', annot_kws=annot_kws)
    print("Test Set Precision: {0:.2%}".format(cm[1,1]/(cm[0,1]+cm[1,1])))
    print("Test Set Recall: {0:.2%}".format(cm[1,1]/(cm[1,1]+cm[1,0])))
    print("Test Set Accuracy: {0:.2%}".format(model.score(X_test, y_test)))