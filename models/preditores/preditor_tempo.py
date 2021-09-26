from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor

# função para treinar o modelo
def train_model_tempo(x, y, categoriza_t_int=False):#data, variavel_predita, variaveis_dependentes):


    #data = get_data()
    # retira as variáveis dependentes
        #x = data.drop(variaveis_dependentes, axis=1) # axis=1 (columns), axis=0(index), default 0
    # informa a variável dependente a ser prevista
        #y = data[variavel_predita]
    
    

    # RandomForestRegressor 
    #model = RandomForestRegressor(random_state=1)
    #params = {
    #           "n_estimators": [50, 100, 200, 400],     #50
    #           "criterion": ['mse', 'mae'],             #mae
    #           "min_samples_split": [2, 4, 8],          #2
    #           "min_samples_leaf": [1, 2, 3],           #3
    #           "min_weight_fraction_leaf": [0.0, 0.1],  #0.0
    #           "max_features": [3, 'auto', 'sqrt'],     #3
    #           "min_impurity_decrease": [0.0, 0.1]      #0.0
    #          }
    #
    #rf_regressor = RandomForestRegressor(
    #    random_state=1,
    #    n_estimators=50,
    #    criterion='mae',
    #    min_samples_split=2,
    #    min_samples_leaf=3,
    #    min_weight_fraction_leaf=0.0,
    #    max_features=3,
    #    min_impurity_decrease=0.0
    #)
    #model = rf_regressor.fit(x, y)

    # RandomForestClassifier
    #model = RandomForestClassifier(random_state=1)
    #params = {
    #           "n_estimators": [50, 100, 200, 400],     #50
    #           "criterion": ['mse', 'mae'],             #mae
    #           "min_samples_split": [2, 4, 8],          #2
    #           "min_samples_leaf": [1, 2, 3],           #3
    #           "min_weight_fraction_leaf": [0.0, 0.1],  #0.0
    #           "max_features": [3, 'auto', 'sqrt'],     #3
    #           "min_impurity_decrease": [0.0, 0.1]      #0.0
    #          }
    #
    #classifier = RandomForestClassifier(
    #    random_state=1
    #)
    #model = classifier.fit(x, y)

    # GradientBoostingRegressor
    #model = GradientBoostingRegressor(random_state=1)
    #params = {
    #            "n_estimators": [50, 100, 200, 400], #100
    #            "loss": ['ls', 'lad', 'huber', 'quantile'], #lad
    #            "learning_rate": [0.1, 0.2], #0.2
    #            "subsample": [0.5, 0.8, 1.0], #0.8
    #            "criterion": ['friedman_mse', 'mse'], #friedman_mse #'mae' is deprecated. The correct way of minimizing the absolute error is to use loss='lad' instead.
    #            "min_samples_split": [2, 4, 8], #2
    #            "min_samples_leaf": [1, 2, 3], #1
    #            "min_weight_fraction_leaf": [0.0, 0.1], #0.1
    #            "min_impurity_decrease": [0.0, 0.1], #0.1
    #            "max_features": [3, 'auto', 'sqrt'], #3
    #            "ccp_alpha": [0.0, 0.1] #0.0
    #          }
    #
    gb_regressor = GradientBoostingRegressor(
        random_state=1,
        n_estimators=100,
        loss='lad',
        learning_rate=0.2,
        subsample=0.8,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.1,
        min_impurity_decrease=0.1,
        max_features=3,
        ccp_alpha=0.0
    ) 
    model = gb_regressor.fit(x, y)                                      
     
    #gs = GridSearchCV(model, param_grid=params, n_jobs=-1, verbose=1)
    #gs.fit(x, y)
    #print("Best params: ", gs.best_params_)
    #model = gs.best_estimator_ 

    return model   
          


############################################### METHODS RANDOM FOREST REGRESSOR ###################################################
#
#   apply(X)
#   Apply trees in the forest to X, return leaf indices.
#
#   decision_path(X)
#   Return the decision path in the forest.
#
#   fit(X, y[, sample_weight])
#   Build a forest of trees from the training set (X, y).
#
#   get_params([deep])
#   Get parameters for this estimator.
#
#   predict(X)
#   Predict regression target for X.
#
#   score(X, y[, sample_weight])
#   Return the coefficient of determination R^2 of the prediction.
#
#   set_params(**params)
#   Set the parameters of this estimator.
####################################################################################################################################


