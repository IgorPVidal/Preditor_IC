from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

import numpy as np
import math

# https://scikit-learn.org/stable/modules/svm.html#svm-classification

def train_model_obito(x, y):#data, variavel_predita, variaveis_dependentes
    #data = get_data()
    # retira as variáveis dependentes
        #x = data.drop(variaveis_dependentes, axis=1) # axis=1 (columns), axis=0(index), default 0
    # informa a variável dependente a ser prevista
        #y = data[variavel_predita]
        
    #Prefer dual=False when n_samples > n_features.
    #When warm_start set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution
    # No momento, todos os valores passados por parâmetro em LogisticRegression correspondem ao valor "default", 
    # e estão sendo passados explicitamente apenas para ajudar a recordar os parâmetros existentes.                                                                                    max_iter estava 800
    
    # GradientBoostingClassifier
    #classifier = GradientBoostingClassifier(random_state=1)
    #params = {
    #    "loss": ['deviance', 'exponential'],
    #    "learning_rate": [0.1, 0.2],
    #    "n_estimators": [50, 100, 200, 400],
    #    "subsample": [0.5, 1.0],
    #    "criterion": ['friedman_mse', 'mse'],
    #    "min_samples_split": [2, 4, 8],
    #    "min_samples_leaf": [1, 2, 3],
    #    "min_weight_fraction_leaf": [0.0, 0.1],
    #    "min_impurity_decrease": [0.0, 0.1],
    #    "max_features": [1, 2, 3, 4, 5, 'auto'],
    #}

    # DecisionTreeClassifier
    #classifier = DecisionTreeClassifier(random_state=1)
    #params = {
    #    "criterion": ['gini', 'entropy'],
    #    "splitter": ['best', 'random'],
    #    "min_samples_split": [2, 4, 8],
    #    "min_samples_leaf": [1, 2, 3],
    #    "min_weight_fraction_leaf": [0.0, 0.1],
    #    "max_features": [1, 2, 3, 4, 5, 'auto'],
    #    "min_impurity_decrease": [0.0, 0.1]
    #}
    #
    #model = DecisionTreeClassifier(
    #    random_state=1,
    #    criterion='entropy',
    #    max_features=1,
    #    min_impurity_decrease=0.0,
    #    min_samples_leaf=2,
    #    min_samples_split=8,
    #    min_weight_fraction_leaf=0.0,
    #    splitter='best'
    #)

    # KNeighborsClassifier
    #classifier = KNeighborsClassifier(n_jobs=-1)
    #params = {
    #    "n_neighbors": [3, 5, 10, 15, 20],
    #    "weights": ['uniform','distance'],
    #    "algorithm": ['ball_tree', 'kd_tree', 'brute'],
    #    #"leaf_size": [1, 5, 10, 20, 30, 40, 50, 100],
    #    "p": [1, 2, 3]
    #}
    #
    #model = KNeighborsClassifier(
    #    algorithm='ball_tree',
    #    n_neighbors=10,
    #    p=1,
    #    weights='distance'
    #)


    # RandomForestClassifier
    #classifier = RandomForestClassifier(random_state=1, n_jobs=-1)
    #params = {
    #    "n_estimators": [50, 100, 200, 400],
    #    "criterion": ['gini', 'entropy'],
    #    "min_samples_split": [2, 4, 8], 
    #    "min_samples_leaf": [1, 2, 3],
    #    "min_weight_fraction_leaf": [0.0, 0.1],
    #    #"max_features": [1, 2, 3, 4, 5, 'auto'],
    #    "max_features": [1, 2, 3, 4],# 5, 'auto'],
    #    "min_impurity_decrease": [0.0, 0.1],
    #    #"max_samples": [round(math.sqrt(x.shape[0]))]#0.2, 0.5, 0.8],
    #}
    #
    model = RandomForestClassifier(
        random_state=1,
        criterion='gini',
        max_features=1,
        min_impurity_decrease=0.0,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        n_estimators=400#1500 (melhor até agora combinando com os parametros achados no gridsearch para False,False,False)
    )

    # SVM
    #classifier = svm.SVC(random_state=1)
    #params = {
    #    #"kernel": ['linear', 'rbf', 'poly'], #'linear'
    #    "kernel": ['rbf'],
    #    "gamma": np.logspace(-9, 3, 13), # 0.1
    #    #"C": [0.1, 0.5, 0.8, 1.0, 2.0, 10.0] #10
    #    "C": np.logspace(-2, 10, 13),
    #    #"degree": [2, 3, 4]
    #}

    # LogisticRegression
    #classifier = LogisticRegression(random_state=1, n_jobs=-1)#n_jobs=-1, random_state=1)
    #params = {
    #            #"penalty": ['l1', 'l2', 'elasticnet', 'none'],
    #            "penalty": ['l2'],
    #            #"C": [0.5, 0.8, 1.0],
    #            "C": np.logspace(-2, 10, 13),
    #            "fit_intercept": [True, False],
    #            #"intercept_scaling": [0.5, 1.0, 1.5],                              #Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True.
    #            #"solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    #            "solver": ['lbfgs'],
    #            #"max_iter": [50, 100, 500, 1000],
    #            "max_iter": [50, 100, 500, 1000],
    #            #"multi_class": ['ovr', 'multinomial'],
    #            "multi_class": ['ovr', 'multinomial'],
    #            #"l1_ratio": [0.0, 0.2, 0.5, 0.8, 1.0]                                        #Only used if penalty='elasticnet'
    #          }

   
    #gs = GridSearchCV(classifier, param_grid=params, scoring=['f1_weighted'], refit='f1_weighted', n_jobs=-1, verbose=1)
    #gs.fit(x, y)
    #print("Best params: ", gs.best_params_)
    #model = gs.best_estimator_    


    #model = LogisticRegression(
    #              random_state=1,
    #              C=100000.0, 
    #              fit_intercept=True,
    #              max_iter=100, 
    #              multi_class='ovr', 
    #              penalty='l2',  
    #              solver='lbfgs'
    #            )
    
    #model = svm.SVC(
    #                random_state=1,
    #                kernel='rbf', 
    #                gamma=1e-05,
    #                C=1000000.0
    #           )

    model.fit(x, y)

    return model

#####################################################################################################################################
# FONTE: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression                                              #
#                                                                                                                                   #
# The "solver" is the Algorithm to use in the optimization problem.                                                                 #
#                                                                                                                                   #
# The solvers implemented in the class LogisticRegression are “liblinear”, “newton-cg”, “lbfgs”, “sag” and “saga”:                  #
#                                                                                                                                   #
# The solver “liblinear” uses a coordinate descent (CD) algorithm, and relies on the excellent C++ LIBLINEAR library,               #
# which is shipped with scikit-learn. However, the CD algorithm implemented in liblinear cannot learn a true multinomial            #
# (multiclass) model; instead, the optimization problem is decomposed in a “one-vs-rest” fashion so separate binary classifiers     #
# are trained for all classes. This happens under the hood, so LogisticRegression instances using this solver behave as multiclass  #
# classifiers. For regularization sklearn.svm.l1_min_c allows to calculate the lower bound for C in order to get a non “null”       #
# (all feature weights to zero) model.                                                                                              #
#                                                                                                                                   #
# The “lbfgs”, “sag” and “newton-cg” solvers only support regularization or no regularization, and are found to converge faster     #
# for some high-dimensional data. Setting multi_class to “multinomial” with these solvers learns a true multinomial                 #
# logistic regression model 5, which means that its probability estimates should be better calibrated than the default              #
# “one-vs-rest” setting.                                                                                                            #
#                                                                                                                                   #
# The “sag” solver uses Stochastic Average Gradient descent 6. It is faster than other solvers for large datasets,                  #
# when both the number of samples and the number of features are large.                                                             #
#                                                                                                                                   #
# The “saga” solver 7 is a variant of “sag” that also supports the non-smooth penalty="l1". This is therefore the solver of choice  #
# for sparse multinomial logistic regression. It is also the only solver that supports penalty="elasticnet".                        #
#                                                                                                                                   #
# The “lbfgs” is an optimization algorithm that approximates the Broyden–Fletcher–Goldfarb–Shanno algorithm 8, which belongs        #
# to quasi-Newton methods. The “lbfgs” solver is recommended for use for small data-sets but for larger datasets its performance    #
# suffers. 9                                                                                                                        #
#####################################################################################################################################

