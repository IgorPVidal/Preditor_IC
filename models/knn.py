import math 
import statistics 

#class KNN:
 #   def __init__(self, data, num_k_vizinhos, lamb):
  #      self.data = data
   #     self.k = num_k_vizinhos
    #    self.lamb = lamb
     #   pass

# Função para completar dados faltantes
# Utilizando KNN (K-Nearest Neighbors)
def completa_dados_faltantes(data, k, lamb=99999999):
    #print("LAMBDA: ", lamb)
    missing_values = data.isna()
    num_linhas = data.shape[0]
    num_colunas = data.shape[1]
    distancia = 0
    k_vizinhos = []
    # Para cada tupla
    for i in range(num_linhas):
        # Descobre se há valor faltante
        for j in range(num_colunas):
            if missing_values.iat[i, j]:
                # Se houver, calcula o valor considerando os k_vizinhos
                if len(k_vizinhos) == 0:
                    # Encontra os k vizinhos mais próximos
                    k_vizinhos = encontra_knn(data, i, j, k, lamb) # O j serve para ver se o valor que vai ser substituido é nan no vizinho
                data.iat[i, j] = calcula_valor_substituto(data, j, k_vizinhos)
        k_vizinhos = []

def calcula_valor_substituto(data, j, vizinhos):
    valores = []
    #print(vizinhos)
    for i in range(len(vizinhos)):
        index_vizinho = vizinhos[i]["posicao"]
        valores.append(data.iat[index_vizinho, j])
    #print("------------------------------- Valores -------------------------------")
    #print(valores)
    data_types = data.dtypes.index
    #print("dtyyyyyyyyy: ", data_types)
    # Testa se deve ser utilizado mediana ou moda
    #print("tipo de dado: ", data_types[j])
    if data_types[j] in ['IDADE', 'DIF_T_PROT_X_SINT', 'DIF_ATB_X_SINT', 'T_INT']:
        #print("Mediana: ", statistics.median(valores))
        return statistics.median(valores)
    else:
        #print("Moda: ", statistics.mode(valores))
        return statistics.mode(valores)

# Encontra os k vizinhos mais próximos do elemento da tupla correspondente à linha "a"
def encontra_knn(data, a, j, k, lamb):
    # Lista para armazenar os vizinhos. Cada vizinho será representado por um dicionario que contém "posicao" e "distancia"
    vizinhos = []
    num_linhas = data.shape[0]
    for b in range(num_linhas):
        # Só deve considerar adicionar à lista de vizinhos se o valor que será usado no calculo não for nan
        if not math.isnan(data.iat[b, j]):
            distancia = distancia_knn(data, a, b, lamb)
            if len(vizinhos) < k:
                vizinhos.append({"posicao":b, "distancia":distancia})
            else:
                substitui_distante(vizinhos, b, distancia)
    return vizinhos

# Determina a distância entre o elemento da linha "a" e o da linha "b"
def distancia_knn(data, a, b, lamb):
    num_colunas = data.shape[1]
    distancia = 0
    for i in range(num_colunas):
        if not math.isnan(data.iat[a, i]) and not math.isnan(data.iat[b, i]):
            # Soma o módulo da subtração
            # OBSERVAÇÃO: NAO ESTÁ FAZENDO IGUAL AO EXEMPLO NESTE PARTE. NÃO ENTENDI COMO FUNCIONA AQUELA FUNÇÃO F
            #               NAO SEI COMO DEVERIA FUNCIONAR PARA VARIAVEIS CATEGORICAS
            distancia += math.fabs(data.iat[a, i] - data.iat[b, i]) #fixme implementar função F() do artigo
        else: 
            distancia += lamb # TALVEZ 0.5 NAO SEJA O MELHOR LAMBDA PARA O JEITO QUE A FUNCAO ESTA AGORA. OS VALORES NAO SAO ENTRE 0 E 1
    if math.isnan(distancia):
        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-DISTANCIA NAN: ", distancia)
    return distancia

# Se a distancia do vizinho testado for menor do que algum da lista de vizinhos,
# Substitui o mais distante
def substitui_distante(vizinhos, b, distancia):
    # inicia assumindo que o elemento da posicao "b" é o mais distante
    maior_distancia = distancia
    # Armazena a posicao a ser substituída na lista. Se chegar ao final com o valor "-1", significa que "b" não está 
    # mais próximo do que os vizinhos presentes na lista
    pos_lista = -1
    for i in range(len(vizinhos)):
        if vizinhos[i]["distancia"] > maior_distancia:
            maior_distancia = vizinhos[i]["distancia"]
            pos_lista = i
    if pos_lista != -1:
        vizinhos[pos_lista]["posicao"] = b 
        vizinhos[pos_lista]["distancia"] = distancia

            