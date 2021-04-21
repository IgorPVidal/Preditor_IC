import streamlit as st
import plotly.express as px 
# talvez usar tkinter?
#import tkinter as tk
#from tkinter import ttk

# OBSERVAÇÃO: Aparentemente Streamlit não reconhece algumas tipagens do Pandas, como 'category'

# no exemplo herdou de tkinter. devo fazer o mesmo com streamlit?
class View():
    def __init__(self, controller):
        self.controller = controller
        self.monta_pagina()
        pass
    
    def monta_pagina(self):
        # pega o conjunto de dados do Model através do Controller
        data = self.controller.get_data()

        # botão para gerar relatorio do pandas_profiling com os dados iniciais
        self.btn_pandas_profiling_inicial = st.button("Gerar relatório do pandas_profiling com dados iniciais")
        # botão para gerar relatorio do pandas_profiling
        self.btn_pandas_profiling = st.button("Gerar relatório do pandas_profiling com dados tratados")
        # título
        st.title("Data App - Previsao de óbito e tempo de internação")
        # subtítulo
        st.markdown("Este é um Data App utilizado para exibir a previsao de óbito e tempo de internação")
        # verificando o dataset
        st.subheader("Selecionando apenas um pequeno conjunto de atributos")
        # atributos para serem exibidos por padrão
        defaultcols = ["IDADE","OBI","T_INT"]
        # defindo atributos a partir do multiselect
        cols = st.multiselect("Atributos", data.columns.tolist(), default=defaultcols)
        # exibindo os top 10 registro do dataframe
        st.dataframe(data[cols].head(10))
        # estatistica descritiva das variáveis
        st.subheader("estatistica descritiva das variáveis")
        st.dataframe(data[st.multiselect("Atributos", data.columns.tolist(), default=["T_INT", "OBI"])].describe())
        
        # distribuicao de pacientes
        st.subheader("Distribuição de pacientes por tempo para internação")
        # definindo a faixa de tempo
        faixa_valores = st.slider("Faixa de tempo", float(data.T_INT.min()), 150., (-100.0, 100.0))

        # filtrando os dados
        dados = data[data['T_INT'].between(left=faixa_valores[0],right=faixa_valores[1])]
        # plota a distribuição dos dados
        f = px.histogram(dados, x="T_INT", nbins=100, title="Distribuição de tempo de internação")
        f.update_xaxes(title="T_INT")
        f.update_yaxes(title="Total Pacientes")
        st.plotly_chart(f)

        st.sidebar.subheader("Defina os atributos do paciente para predição")
        ############################################################################################
        ## mapeando dados do usuário para cada atributo ############################################
        # fixme generalizar o que possivel (talvez nao seja possivel nesse caso)
        self.idade = st.sidebar.number_input("IDADE", value=data.IDADE.mean())
        self.dif_t_prot_x_sint = st.sidebar.number_input("DIF_T_PROT_X_SINT", value=data.DIF_T_PROT_X_SINT.mean())
        self.dif_atb_x_sint = st.sidebar.number_input("DIF_ATB_X_SINT", value=data.DIF_ATB_X_SINT.mean())
        self.sem_foco = st.sidebar.checkbox("SEM_FOCO") 
        self.mucosas = st.sidebar.checkbox("F_MUCOSAS")
        self.pulmao = st.sidebar.checkbox("F_PULMAO")
        self.renal = st.sidebar.checkbox("F_RENAL")
        self.atb = st.sidebar.selectbox("ATB (Antibióticos)", (0, 1, 2, 3, 4, 5))
        self.dva = st.sidebar.selectbox("DVA (Drogas Vasoativas)", (0, 1, 2, 3))
        self.ant_pes = st.sidebar.selectbox("ANT_PES", (0, 1, 2, 3))
        ###########################################################################################
        # botão para realizar a predição
        self.btn_predict = st.sidebar.button("Realizar Predição")
        pass

    def sucesso_pandas_profiling_inicial(self):
        st.subheader("O arquivo Relatorio_dados_iniciais.html deveria agora se encontrar na mesma pasta do appH.py")
        pass

    def sucesso_pandas_profiling(self):
        st.subheader("O arquivo Relatorio_dados_tratados.html deveria agora se encontrar na mesma pasta do appH.py")
        pass

    def mostra_predicoes(self, predicoes):
        st.subheader("O tempo de internação previsto é:")
        st.write("RandomForest: ")
        st.write(predicoes[0])

        st.subheader("Previsão de óbito:")
        st.write("Regressão Binária: ")
        st.write(predicoes[1])
        if predicoes[1] == 1:
            st.write("(sim)")
            pass
        else:
            st.write("(não)")
            pass


