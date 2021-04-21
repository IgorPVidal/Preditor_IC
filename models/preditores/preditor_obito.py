from sklearn.linear_model import LogisticRegression

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
    
    logistic_regressor = LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=1000, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
    logistic_regressor.fit(x, y)
    return logistic_regressor

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

