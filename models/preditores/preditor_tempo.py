from sklearn.ensemble import RandomForestRegressor 

# função para treinar o modelo
def train_model_tempo(x, y):#data, variavel_predita, variaveis_dependentes):
    #data = get_data()
    # retira as variáveis dependentes
        #x = data.drop(variaveis_dependentes, axis=1) # axis=1 (columns), axis=0(index), default 0
    # informa a variável dependente a ser prevista
        #y = data[variavel_predita]
    rf_regressor = RandomForestRegressor(n_estimators=200, max_depth=7, max_features=3)

    #o GridSearchCV que deve fazer o fit
    # verbose avisa quantos testes vão ocorrer, para sabermos se vai demorar 
    
    rf_regressor.fit(x, y)
    return rf_regressor

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


