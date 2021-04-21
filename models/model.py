import pandas as pd 


from scipy import stats
from pandas_profiling import ProfileReport

from sklearn.impute import SimpleImputer #teste
from sklearn.impute import KNNImputer    #teste
from sklearn.preprocessing import StandardScaler #teste
from sklearn.feature_selection import VarianceThreshold #teste
from sklearn.model_selection import train_test_split #teste
from sklearn.metrics import classification_report #teste }
from sklearn.metrics import confusion_matrix      #teste }
from sklearn.model_selection import cross_val_score #teste }
from sklearn.model_selection import GridSearchCV  #teste }
#from sklearn import svm
import numpy as np

if __name__ == "__main__":
    from knn import completa_dados_faltantes
    from preditores import preditor_obito
    from preditores import preditor_tempo
    import math
else:
    from models.knn import completa_dados_faltantes
    from models.preditores import preditor_obito
    from models.preditores import preditor_tempo

class Model:
    def __init__(self, caminho_csv): 
        self.dtypes = { #considerar fazer um "dados.py" para definir o dicionario dtypes, e o Model o importa
            'IDADE': 'int64',
            'DIF_T_PROT_X_SINT': 'float64',
            'DIF_ATB_X_SINT': 'float64',   
            'SEM_FOCO': 'bool',
            'F_ABDOMINAL': 'bool',
            'F_CORACAO': 'bool',
            'F_GENITAL': 'bool',
            'F_INFECCAO_DE_FERIDO_OPERATORIA': 'bool',
            'F_INTESTINAL': 'bool',
            'F_MUCOSAS': 'bool',
            'F_OSSEO': 'bool',
            'F_PELE': 'bool',
            'F_PULMAO': 'bool',
            'F_RENAL': 'bool',
            'F_SANGUE': 'bool',
            'ATB': 'category',
            'DVA': 'category',
            'OBI': 'bool',
            'T_INT': 'float64',
            'ANT_PES': 'category'
        }
        self.bool_dtypes = []
        self.category_dtypes = []
        self.numeric_dtypes = []
        for column in self.dtypes:
            if self.dtypes[column] == 'bool':
                self.bool_dtypes.append(column)
            elif self.dtypes[column] == 'category':
                self.category_dtypes.append(column)
            else:
                self.numeric_dtypes.append(column)
        #print(self.bool_dtypes)
        #print(self.category_dtypes)
        #print(self.numeric_dtypes)

        self.data_inicial = self.get_data(caminho_csv) # Armazena uma cópia dos dados antes do tratamento
        self.data = self.get_data(caminho_csv)

        # Separa os dados por tipo
        self.bool_data = self.data.drop(self.category_dtypes + self.numeric_dtypes, axis=1)
        self.category_data = self.data.drop(self.bool_dtypes + self.numeric_dtypes, axis=1)
        self.numeric_data = self.data.drop(self.bool_dtypes + self.category_dtypes, axis=1)
        #print(self.bool_data)
        #print(self.category_data)
        #print(self.numeric_data)
        #print(self.data)

        # tratar os dados antes de treinar modelo
        self.trata_dados()

        #miss_values = self.data.isna()
        #ha_valor_faltante = False
        #for i in range(self.data.shape[0]):
        #    for j in range(self.data.shape[1]):
        #        if miss_values.iat[i,j]:
        #            ha_valor_faltante = True
        #if ha_valor_faltante:
        #    print("HA VALORES FALTANTES--------------------------")
        #else: 
        #    print("NENHUM VALOR FALTANTE!!!!!!!--------------------------")

        #self.knn = KNN(self.data, num_k_vizinhos=10, lamb=99999999)
        # treinando o modelo para tempo de internação
        
        self.model_tempo = preditor_tempo.train_model_tempo(self.X_train_int, self.y_train_int)#data=self.data, variavel_predita="T_INT", variaveis_dependentes=["OBI", "T_INT"])
        # treinando o modelo para obito
        self.model_obito = preditor_obito.train_model_obito(self.X_train_obi, self.y_train_obi)#data=self.data, variavel_predita="OBI", variaveis_dependentes=["OBI", "T_INT"])
        pass

    # função para carregar o dataset
    #@st.cache # para ficar no cache do streamlit e acelerar o processo
    def get_data(self, caminho_csv):
        return pd.read_csv(caminho_csv)

    def trata_dados(self):
        #print("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")
        #print(self.data.dtypes)
        data_copy = self.data.copy(deep=True)
        
        #print("NUMERIC 1")
        #print(self.numeric_data)
        #print("BOOL 1")
        #print(self.bool_data)
        #print("CATEGORY 1")
        #print(self.category_data)
        # ------- DATA IMPUTATION ------- 
                    #USAR KNN PARA NUMERICOS E SIMPLEIMPUTER PARA BOOL E CATEGORY (?)
        # Usando KNN para imputar dados numéricos
        imp = KNNImputer(n_neighbors=5)
        imputed_data = pd.DataFrame(data=imp.fit_transform(data_copy), index=data_copy.index, columns=data_copy.columns) #imputando dados com knn somente no numeric_data, o R2_SCORE de OBI aumentou 0.5%
        self.numeric_data = imputed_data.drop(self.bool_dtypes + self.category_dtypes, axis=1) 
        # Usando SimpleImputer para imputar dados booleanos e categóricos
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent') # Irá preencher com a moda
        imputed_data = pd.DataFrame(data=imp.fit_transform(data_copy), index=data_copy.index, columns=data_copy.columns)
        self.bool_data = imputed_data.drop(self.category_dtypes + self.numeric_dtypes, axis=1)
        self.category_data = imputed_data.drop(self.bool_dtypes + self.numeric_dtypes, axis=1)

        
        #print("NUMERIC 2")
        #print(self.numeric_data)
        #print("BOOL 2")
        #print(self.bool_data)
        #print("CATEGORY 2")
        #print(self.category_data)
        #print("Resultante")
        #print(self.data)

        #self.data = 
        # ------- DATA PREPROCESSING -------
        #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
                #scaler = StandardScaler(with_mean=False, with_std=True)
        #print("aaaaaaaaaaaaaaaa")
        #print(self.numeric_data)
        #print("bbbbbbbbbbbbbbbb")
                #self.numeric_data = pd.DataFrame(data=scaler.fit_transform(self.numeric_data), index=self.numeric_data.index, columns=self.numeric_data.columns)
        
        #print(self.numeric_data)
        #self.corrige_categorico()
        #print(self.data)
        #print("cccccccccccccccc")
        #print(self.data.dtypes)

        # ------- FEATURE SELECTION -------
        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
        #print("--------------- num 1 ---------------")
        #print(self.numeric_data)
        #print("--------------- bool 1 ---------------")
        #print(self.bool_data)
        #print("--------------- cat 1 ---------------")
        #print(self.category_data)
        
        selector = VarianceThreshold(threshold=0.1)

        self.numeric_data = selector.fit_transform(self.numeric_data)
        removidos = selector.get_support()
        #print(removidos)
        count = 0
        dtypes_copy = self.numeric_dtypes.copy()
        for dtype in dtypes_copy:
            if not removidos[count]:
                self.numeric_dtypes.remove(dtype)
                self.dtypes.pop(dtype)
            count += 1
        #print(self.numeric_dtypes)
        self.numeric_data = pd.DataFrame(data=self.numeric_data, index=range(self.numeric_data.shape[0]), columns=self.numeric_dtypes)

        self.bool_data = selector.fit_transform(self.bool_data)
        removidos = selector.get_support()
        #print(removidos)
        count = 0
        dtypes_copy = self.bool_dtypes.copy()
        for dtype in dtypes_copy:
            if not removidos[count]:
                self.bool_dtypes.remove(dtype)
                self.dtypes.pop(dtype)
            count += 1
        #print(self.bool_dtypes)
        self.bool_data = pd.DataFrame(data=self.bool_data, index=range(self.bool_data.shape[0]), columns=self.bool_dtypes)

        self.category_data = selector.fit_transform(self.category_data)
        removidos = selector.get_support()
        #print(removidos)
        count = 0
        dtypes_copy = self.category_dtypes.copy()
        for dtype in dtypes_copy:
            if not removidos[count]:
                self.category_dtypes.remove(dtype)
                self.dtypes.pop(dtype)
            count += 1
        #print(self.category_dtypes)
        self.category_data = pd.DataFrame(data=self.category_data, index=range(self.category_data.shape[0]), columns=self.category_dtypes)
        
        #print("dtypes:")
        #print(self.dtypes)

        #print("--------------- num 2 ---------------")
        #print(self.numeric_data)
        #print("--------------- bool 2 ---------------")
        #print(self.bool_data)
        #print("--------------- cat 2 ---------------")
        #print(self.category_data)

        #POSSO TRATAR TUDO SEPARADAMENTE E DEPOIS PASSAR PRO self.data ASSIM LÁÁÁÁÁ NO FINAL (?)
        self.data = pd.DataFrame()
        for column in self.numeric_data:
            self.data[column] = self.numeric_data[column]
        for column in self.bool_data:
            self.data[column] = self.bool_data[column]
        for column in self.category_data:
            self.data[column] = self.category_data[column]
        self.data = self.data.astype(self.dtypes)
        #print(self.data)
    
        ############################################self.corrige_categorico()
        #print(self.data)

        # Considerar fazer um "tratamento separado" para cada tipo de predição.
        # Pois, em alguns casos, a melhora do R2 Score de uma coincide com a piora da outra.
        # Por isso, talvez seja bom otimizar as duas separadamente.
        self.data.drop(
            [
                #"DIF_T_PROT_X_SINT",
                #"DIF_ATB_X_SINT",

                #"SEM_FOCO", 
                #"F_ABDOMINAL", 
                #"F_CORACAO", 
                #"F_GENITAL", 
                #"F_INFECCAO_DE_FERIDO_OPERATORIA", 
                #"F_INTESTINAL", 
                #"F_OSSEO", 
                #"F_SANGUE"
            ], 
            axis=1, inplace=True
        )
                #self.merge_colunas(["F_MUCOSAS","F_PELE"]) # Esta função está tornando as duas colunas iguais. Se uma das duas tem uma linha com valor 1, então as duas terão 1 nessa linha.
        #self.data.drop(["F_MUCOSAS", "F_PELE"], axis=1, inplace=True)
                #self.data.drop(["F_PELE"], axis=1, inplace=True)
        # Para que os valores da coluna ATB sejam vistos como categóricos
        #print(self.data.dtypes)
        #print(self.data.ATB.astype('category'))
        #self.data.ATB = self.data.ATB.astype('category') 
        #print(self.data.dtypes)
        #completa_dados_faltantes(self.data, k=10, lamb=99999999) #lamb = 99999999 #0.5    #fixme
        #fixme O fit é realizado, o array volta a ser um DataFrame, mas os tipos dos dados se tornam todos float
        #self.preenche_nan()
        #print(self.data.dtypes)                                                    
        
        #imp.fit(self.data.to_numpy())
        
        #print(self.data.dtypes)
        
        #print(self.data.dtypes)
        
                #svc = svm.SVC()
                #clf = GridSearchCV(svc, [])
        
        #print(self.data)
        # ------- TRAINING (NÃO SERÁ FEITO AQUI) -------
        # Separa conjunto de treinamento e de testes para a variável "OBI"
        self.X_train_obi, self.X_test_obi, self.y_train_obi, self.y_test_obi = train_test_split(self.data.drop(["T_INT","OBI"],axis=1,inplace=False), self.data["OBI"], test_size=0.33, random_state=42, stratify=self.data["OBI"])
        # Separa conjunto de treinamento e de testes para a variável "T_INT"
        self.X_train_int, self.X_test_int, self.y_train_int, self.y_test_int = train_test_split(self.data.drop(["T_INT","OBI"],axis=1,inplace=False), self.data["T_INT"], test_size=0.33, random_state=42)

        #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html }
        #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html      }
        #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html }
        #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html    }

    #fixme talvez implementar esse método para automatizar o processo de alteração de dtypes,
    #        sem necessidade de dizer explicitamente quais são. Tentar fazer conferir automaticamente.
    #def corrige_dtypes(self, dtypes):
     #   self.data.IDADE.astype("int64")
     #   #self.data.DIF_T_PROT_X_SINT.astype(dtypes[1])
     #   #self.data.DIF_ATB_X_SINT.astype(dtypes[2])
     #   self.data.F_MUCOSAS.astype("int64")
     #   self.data.F_PULMAO .astype("int64")
     #   self.data."F_RENAL".astype("int64")
     #   self.data.ATB.astype("category")
     #   self.data.DVA.astype("category")
     #   self.data.OBI.astype("int64")
     #   #self.data.T_INT.astype(dtypes[9])
     #   self.data.ANT_PES.astype("category")

                    
                
    # no momento usado apenas para fazer com que as colunas F_PELE e F_MUCOSAS representem a mesma informação
    # para remover uma das duas. Pois a correlação é alta e positiva, e possivelmente representam a mesma coisa 
    # apenas preenchida de forma diferente
    # por enquanto só funciona para valores binários
    def merge_colunas(self, colunas):
        num_linhas = self.data.shape[0]
        coluna1 = colunas[0]
        coluna2 = colunas[1]
        # Para cada tupla
        for i in range(num_linhas):
            # Confere se uma das duas colunas possui um valor "1"
            if (self.data.at[i, coluna1] == 1 or self.data.at[i, coluna2] == 1):
                #print("-----________________________ 'MERGE'!!! __________________________---------------")
                #print(coluna1, ": ", self.data.at[i, coluna1], "   ", coluna2, ": ", self.data.at[i, coluna2])
                
                # Se sim, as duas terão o valor "1"
                self.data.at[i, coluna1] = 1
                self.data.at[i, coluna2] = 1

    def coeficientePearson(self):
        return stats.skew(self.data.T_INT)

    def score_preditor_tempo(self):
            #x = self.data.drop(["OBI", "T_INT"],axis=1).to_numpy()
            #y_verdadeiro = self.data.filter(["T_INT"]).to_numpy()
        #print("X_test_int")
        #print(self.X_test_int)
        #print("y_true")
        #print(self.y_test_int)
        #print("y_true.mean()")
        #print(self.y_test_int.mean())
        x = self.X_test_int
        y_true = self.y_test_int
        
        
        # classification_report aqui? Parece que que não aceita Y contínuo (float) ValueError: continuous is not supported

        print("========== Cross Validation TEMPO ==========")
        print(cross_val_score(self.model_tempo, self.data.drop(["T_INT","OBI"],axis=1,inplace=False), self.data["T_INT"], cv=3))
        
        


        #print("r2 score Random Forest: ", round(self.model_tempo.score(x, y_verdadeiro)*100, 2), "%")
        return round(self.model_tempo.score(x, y_true)*100, 2)

    def score_preditor_obito(self):
            #x = self.data.drop(["OBI", "T_INT"],axis=1).to_numpy()
            #y_verdadeiro = self.data.filter(["OBI"]).to_numpy()
        x = self.X_test_obi
        y_true = self.y_test_obi#.to_numpy()
        y_pred = self.model_obito.predict(x)

        # classification_report aqui? Parece que que não aceita Y contínuo (float) ValueError: continuous is not supported
        print("========== Classification Report ==========")
        #print(x)
        #print(y_true)
        #print(y_pred)
        print(classification_report(y_true, y_pred, target_names=["True", "False"]))
        print("========== Confusion Matrix ==========")
        #Compute confusion matrix to evaluate the accuracy of a classification.
        #By definition a confusion matrix is such that is equal to the number of observations known to be in group 
        # and predicted to be in group.
        #Thus in binary classification, the count of true negatives is [0][0],
        #false negatives is [1][0], true positives is [1][1] and false positives is [0][1].
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print("True Negatives: ", tn)
        print("False Negatives: ", fn)
        print("True Positives: ", tp)
        print("False Positives: ", fp)

        

        print("========== Cross Validation OBITO ==========")
        print(cross_val_score(self.model_obito, self.data.drop(["T_INT","OBI"],axis=1,inplace=False), self.data["OBI"], cv=3))
        
        
        #print("r2 score Logistic Regressor: ", round(self.model_obito.score(x, y_verdadeiro)*100, 2), "%")
        return round(self.model_obito.score(x, y_true)*100, 2)

    # Gera um relatório com base nos dados iniciais, antes do tratamento
    def gerar_relatorio_inicial(self):
        profile = ProfileReport(self.data_inicial, title='Relatório Inicial - Pandas Profiling', html={'style':{'full_width':True}})
        profile.to_file(output_file="Relatorio_dados_iniciais.html")

    # Gera um relatório com base nos dados tratados
    def gerar_relatorio(self):
        profile = ProfileReport(self.data, title='Relatório - Pandas Profiling', html={'style':{'full_width':True}})
        profile.to_file(output_file="Relatorio_dados_tratados.html")

    def predicao_tempo(self, X):
        result = self.model_tempo.predict(X)
        return result[0]

    def predicao_obito(self, X):
        result = self.model_obito.predict(X)
        return result[0]

    def pega_dados(self):
        return self.data


if __name__ == "__main__":
    print("------------------------- model (TESTES) -------------------------")
    m = Model("../data/dataH3.csv")
    print("=========================  TEMPO  =========================")
    m.score_preditor_tempo()
    #print("valores: ", scores_tempo)
        #print("media: ", round(media_scores_tempo, 2), "%") #fixme ESTÁ NEGATIVO. Conferir todas as mudanças no preditor de tempo para achar o erro
                                                             #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor.score
        #print("desvio padrao: ", round(desvio_padrao_tempo, 4))
    print("=========================  OBITO  ======================")
    m.score_preditor_obito()
    #print("valores: ", scores_obito)
        #print("media: ", round(media_scores_obito, 2), "%")
        #print("desvio padrao: ", round(desvio_padrao_obito, 4))
    pass

    


    
        
    
    


