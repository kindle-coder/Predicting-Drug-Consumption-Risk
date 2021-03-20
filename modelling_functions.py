#Modelling functions


def model_lg(X_train, y_train, X_test):

    #LOGISTIC REGRESSION
    X_train, X_test = scale(X_train, X_test)

    logreg = LogisticRegression(solver = 'liblinear')

    c_space = [0.001,0.01,0.1,1,10,100]
    penalty = ['l1','l2']
    param_grid = {'C': c_space,
             'penalty': penalty}

    clf_logreg = RandomizedSearchCV(logreg, param_grid , cv= 10, scoring ='recall',random_state=30)
    best_model_logreg = clf_logreg.fit(X_train, y_train)

    #print('Best Model Parameters:', best_model_logreg.best_params_)
    print("Best Logistic Regression Score: {0:.2%}".format(best_model_logreg.best_score_))
    return best_model_logreg

def model_dt(X_train, y_train):

    #DECISION TREE
    tree = DecisionTreeClassifier()

    param_dist = {"max_depth": [3, 2, None],
                  "min_samples_leaf": randint(1, 9),
                  "criterion": ["gini", "entropy"]}

    clf_tree = RandomizedSearchCV(tree, param_dist, cv= 10, scoring ='recall', random_state=30)

    best_model_tree = clf_tree.fit(X_train,y_train)

    #print('Best Model Parameters:', best_model_tree.best_params_)
    print("Best Decision Tree Score: {0:.2%}".format(best_model_tree.best_score_))
    return best_model_tree

def model_rf(X_train, y_train):
    
    #RANDOM FOREST
    rf = RandomForestClassifier()

    param_dist = {"n_estimators": randint(4,200),
                "min_samples_split": truncnorm(a=0, b=1, loc=0.25, scale=0.1),
                  "max_depth": [10, 20, 30, 40, 50, 60, 70],

    }

    clf_rf = RandomizedSearchCV(rf, param_dist, cv=10, scoring ='recall',random_state=30)

    best_model_rf = clf_rf.fit(X_train, y_train)

    #print('Best Model Parameters:', best_model_rf.best_params_)
    print("Best Random Forest Score: {0:.2%}".format(best_model_rf.best_score_))
    return best_model_rf

def model_svm(X_train, y_train, X_test):
    
    # SUPPORT VECTOR MACHINE
    X_train, X_test = scale(X_train, X_test)

    rbfSVM = SVC(kernel = 'rbf', probability=True)

    param_dist = { "C": stats.uniform(2, 10),
                 "gamma": stats.uniform(0.1, 1)}


    clf_rbfSVM = RandomizedSearchCV(rbfSVM, param_dist, cv=10, scoring ='recall', random_state=30)

    best_model_rbfSVM = clf_rbfSVM.fit(X_train, y_train)

    #print('Best Model Parameters:', best_model_rbfSVM.best_params_)
    print("Best RBF SVM Score: {0:.2%}".format(best_model_rbfSVM.best_score_))
    return best_model_rbfSVM

def model_all(X_train, y_train, X_test):
    lg = model_lg(X_train, y_train, X_test)
    dt = model_dt(X_train, y_train)
    rf = model_rf(X_train, y_train)
    svm = model_svm(X_train, y_train, X_test)
    return lg, dt, rf, svm