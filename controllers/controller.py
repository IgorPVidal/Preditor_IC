from models.model import Model
from views.view import View

class Controller:
    def __init__(self, caminho_csv):
        self.model = Model(caminho_csv) 
        self.view = View(self)
        pass

    def start(self):
        print("Iniciando.........")
        # verifica se os botões foram acionados
        if self.view.btn_pandas_profiling_inicial:
            self.model.gerar_relatorio_inicial()
            self.view.sucesso_pandas_profiling_inicial()
            pass
        pass
        if self.view.btn_pandas_profiling:
            self.model.gerar_relatorio()
            self.view.sucesso_pandas_profiling()
            pass
        pass
        if self.view.btn_predict:
            X = [[
                self.view.idade, 
                self.view.dif_t_prot_x_sint, 
                self.view.dif_atb_x_sint,
                self.view.sem_foco,
                self.view.mucosas,
                self.view.pulmao, 
                self.view.renal,
                self.view.atb, 
                self.view.dva, 
                self.view.ant_pes
            ]]
            predicoes = [self.model.predicao_tempo(X), self.model.predicao_obito(X)]
            self.view.mostra_predicoes(predicoes)

    def get_data(self):
        return self.model.pega_dados()




