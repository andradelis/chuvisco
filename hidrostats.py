import climsy
import os
import matplotlib.pyplot as plt
import pandas as pd
import re
import pymannkendall as mk
import xarray as xr
import datetime as dt
import numpy as np
import statsmodels.api as sm
from datetime import datetime
from dateutil.relativedelta import relativedelta


class Vazao():
    
    def __init__(self, caminho_ou_dataset, **kwargs):    
        
        if isinstance(caminho_ou_dataset, str):
            vazao = self.lerCSV(caminho = caminho_ou_dataset, **kwargs)
        elif isinstance(caminho_ou_dataset, xr.Dataset):
            vazao = caminho_ou_dataset
        self.vazao = vazao
        

    def agruparMedia(self, freq="MS"):
        """
        Retorna a série temporal após passado o resample de acordo com a frequência escolhida. A sintaxe de frequẽncia segue o padrão do Pandas.
        
        Parâmetros:
        freq: str. Default: "MS"
            Frequência de agrupamento."""
        
        vazao = self.vazao
        
        return Vazao(vazao.resample(time = freq).mean())
        
        
    def lerCSV(self, caminho, **kwargs):
        """
        Lê um arquivo .csv. O comportamento padrão trata o arquivo tal qual o obtido a partir da base de dados da Agência Nacional das Águas.
        
        Kwargs:
        decimal: str
            Representação do decimal no arquivo. Default: ","
        
        index_col: str
            Nome da coluna índice. Default: "Data da Medição"
            
        usecols: str
            Colunas a serem lidas. Default: ["Data da Medição", "Vazão Natural (m³/s)"]"""
        
        self.caminho = caminho        
        vazao = pd.read_csv(self.caminho, decimal = kwargs.get("decimal", ","), index_col=kwargs.get("index_col", "Data da Medição"), usecols=kwargs.get('usecols', ["Data da Medição", "Vazão Natural (m³/s)"]))
        vazao.index = pd.to_datetime(vazao.index, format=kwargs.get('format', '%d/%m/%Y'))

        vazao_ds = xr.Dataset.from_dataframe(vazao)
        vazao_ds = vazao_ds.rename({"Data da Medição": "time"})
        
        return vazao_ds
        
        
    def vazaoPlot(self, ax, **kwargs):
        """
        Plota a série histórica de vazão natural, em m³/s.
        """

        def indiceCategorico(array):
            """
            Converte o vetor temporal em categórico.
            
            Parâmetros:
            array: xr.DataArray
                Vetor temporal.
            """

            def diasConsecutivos(a, b, step = relativedelta(months=1)):
                """
                Checa se duas datas estão separadas por um período de tempo.
                
                Parâmetros:
                a: datetime64[ns]
                    Primeira data.
                    
                b: datetime64[ns]
                    Segunda data.
                    
                step: dateutil.relativedelta
                    Diferença esperada entre as datas.
                """
                return (a + step) == b
            
            
            if all(diasConsecutivos(pd.to_datetime(array[i].values), pd.to_datetime(array[i+1].values)) for i in range(len(array) - 1)):
                pass
            else:
                array["time"] = array["time"].dt.strftime("%d-%m-%Y")

                return array["time"]
                
                
        vazao = self.vazao
        vazao["time"] = indiceCategorico(vazao["time"])
            
        ax.plot(vazao["time"], vazao["Vazão Natural (m³/s)"], label = kwargs.get('legenda', "Vazão natural (SAR/ANA, 2021)"))
        ax.set_ylabel("Vazão Natural\n(m³/s)", fontsize = 13)
            
        if vazao["time"].dtype == 'object':
            start, end = ax.get_xlim()
            ax.xaxis.set_ticks(np.arange(start, end, 10))
            plt.setp(ax.get_xmajorticklabels(), rotation=45, ha="right", fontsize=13)
            
        return ax
            
        
    def recorteTemporal(self, **kwargs):
        """
        Retorna a série temporal recortada. Uma série temporal não linear assume valores categóricos para o vetor de data.
        
        Kwargs
        series: None, str ou list
            Recorte da série temporal. 'rainy season' retorna o recorte apenas no período de Outubro a Março. Uma lista de números inteiros (de 1 a 12) retorna a série nos meses requeridos.
        """
        
        if isinstance(self, Vazao):
            vazao = self.vazao
        elif isinstance(self, xr.Dataset):
            vazao = self
        
        return Vazao(climsy.Climsy.slicer(dataset = vazao,
                                             series = kwargs.get('series', "rainy season")))
    
    
    def massaResidual(self, **kwargs):
        """
        Retorna o diagrama de massa residual (diagrama de Rippl) segundo a equação:
        
        $$\sum^j_{i=n} \frac{Q_{i} - Q_{LTM}}{Q_{LTM}}$$
        
        Considerando Q_{i} a vazão no tempo i e Q_{LTM} a vazão média de longo termo.
        
        Kwargs:
        period: list
            Período no qual está sendo avaliada a anomalia. (Opcional)
        
        basis: list
            Média de longo termo para o cálculo de anomalia. (Opcional)
            
        freq: str. ['month', 'day']
            Frequência da análise. O dado é resampleado em função do parâmetro _freq_. Logo, se _freq_ = 'month', o diagrama retorna a anomalia acumulada mensal. (Opcional)
        """
        
        vazao = climsy.Time(self.vazao)
        rippl_vazao = vazao.rippl(period = kwargs.get('period', ['01-01-2010', '31-12-2018']),
                                  basis = kwargs.get('basis', ['01-01-1979', '31-12-2010']),
                                  freq = kwargs.get('freq', 'month')).dataset
        
        return Vazao(rippl_vazao)
    
    
    def mediaMovel(self, janela):
        """
        Retorna a média móvel da série temporal.
        
        Parâmetros:
        janela: int
            Janela temporal da média móvel.
        """
        
        saida = self.vazao.rolling(time = janela).mean()
        
        return saida
        

    def medianaMovel(self, janela):
        """
        Retorna a mediana móvel da série temporal.
        
        Parâmetros:
        janela: int
            Janela temporal da mediana móvel.
        """

        saida = self.vazao.rolling(time = janela).median()
        
        return saida
    
        
    def regPoli(self, grau_polinomio):
        """
        Retorna um array contendo os valores para a curva de ajuste para a tendência da série segundo o método dos mínimos quadrados.
        
        Parâmetros:
        grau_polinomio: int
            Grau do polinômio de ajuste.
        """
        
        # Obter o vetor de vazão natural
        y = self.vazao["Vazão Natural (m³/s)"]
        # Obter n valores entre 0 e 1, onde n é igual ao comprimento do vetor contendo os valores de vazão natural
        x = np.linspace(0, 1, len(self.vazao["Vazão Natural (m³/s)"]))

        # Obter os coeficientes para o ajuste de um polinômio p(x) = p_0*x^n + p_1*x^(n-1) + ... + p_n, onde n é o grau do polinômio.
        coefs = np.polyfit(x, y, grau_polinomio)
        # Os coeficientes obtidos são parâmetros para a instância da classe de operações polinomiais do numpy
        f = np.poly1d(coefs)
        polinomio = f(x)
        
        return polinomio
    

    def mannKendall(self, ref = "hamed rao"):
        """
        Retorna o teste de Mann-Kendall para a distribuição dos eventos na série histórica. TESTES
        """
        vazao = self.vazao
        
        y = vazao["Vazão Natural (m³/s)"]

        if ref == "hamed rao":
            resultado = mk.hamed_rao_modification_test(y)
                         
        elif ref == "yue wang":
            resultado = mk.yue_wang_modification_test(y)
                         
        elif ref == "trend-free prewhitening":
            resultado = mk.trend_free_pre_whitening_modification_test(y)
                         
        elif ref == "prewhitening":
            resultado = mk.pre_whitening_modification_test(y)
            
        elif ref == 'seasonal':
            resultado = mk.seasonal_test(y, period=6)

        return resultado
    
    def loess(self, **kwargs):
        """
        Retorna uma curva calculada a partir da regressão local da série temporal.
        
        **Kwargs:
        type: 'trend', 'seasonality', 'observed', 'residual'. Default: 'trend'
            Tipos de análise.

        seasonality: int > 3
            Fator de suavização da curva. Ver statsmodels.tsa.seasonal.STL.
                
        period: int
            Ver statsmodels.tsa.seasonal.STL.    
                
        robust: bool
            Ver statsmodels.tsa.seasonal.STL.

        seasonal_deg: int
            Ver statsmodels.tsa.seasonal.STL.

        trend_deg: int
            Ver statsmodels.tsa.seasonal.STL.

        low_pass_deg: int
            Ver statsmodels.tsa.seasonal.STL.
        """
        
        vazao = climsy.Time(self.vazao)
        loess_vazao = vazao.loess(type=kwargs.get('type', 'trend'),
                                  seasonality=kwargs.get('seasonality', 7),
                                  period=kwargs.get('period', 6),
                                  robust=kwargs.get('robust', False),
                                  seasonal_deg=kwargs.get('seasonal_deg', 1), 
                                  trend_deg=kwargs.get('trend_deg', 1), 
                                  low_pass_deg=kwargs.get('low_pass_deg', 1)).dataset
        
        return loess_vazao
    
    def analiseVazao(self, usina, inicio = 1990):
        """
        Compilado de métodos da classe Vazao que facilita a visualização da tendência nas séries históricas de vazão.
        """
        
        def plot(vazao, loess, media, mediana, rippl):
            fig, ax = plt.subplots(nrows=3, figsize = (20, 10))
            vazao.vazaoPlot(ax[0])
            ax[0].plot(loess["Vazão Natural (m³/s)"], label = "Regressão local (CLEVELAND et al., 1988)")
            ax[0].legend()
            ax[0].set_title(f'Vazão natural afluente em {usina}\nMédia mensal de Outubro a Março')

            ax[1].set_ylabel("Variabilidade multidecadal\n(m³/s)", fontsize = 13)
            ax[1].plot(media["Vazão Natural (m³/s)"], label = "Média móvel (10 anos)")
            ax[1].plot(mediana["Vazão Natural (m³/s)"], label = "Mediana móvel (10 anos)")
            ax[1].legend(title="ALVES et al., 2013")
            ax[1].set_xticks([])
            ax[1].set_title(f'Média e mediana móvel para a vazão de {usina} mensal de Outubro a Março')

            ax[2].set_ylabel("Diagrama de massa residual", fontsize = 13)
            ax[2].plot(rippl.vazao["Vazão Natural (m³/s)"], label = "Anomalia acumulada")
            ax[2].plot(rippl.regPoli(3), label = "M.M.Q")
            ax[2].legend()
            ax[2].set_title(f'Diagrama de Ripple a partir das anomalias de vazão natural afluente no reservatório de {usina}\nMédia mensal de Outubro a Março em relação à média de longo termo entre {inicio} e 2010')

            fig.tight_layout()
            
            
        vazao = self.vazao
        
        vazao = vazao.where(vazao['time.year'] >= inicio, drop = True)    
        vazao_media_verao = self.agruparMedia().recorteTemporal()
        
        mk = vazao_media_verao.mannKendall()
        loess = vazao_media_verao.loess()
        # A média e mediana móveis, através de uma janela temporal de 10 anos, descrevem o padrão central de variabilidade de baixa frequência para a série temporal, onde a 
        # mediana é menos influenciada pelos outliers do que a média (ALVES et al., 2013).
        media = vazao_media_verao.mediaMovel(10*6)
        mediana = vazao_media_verao.medianaMovel(10*6)
            
        rippl = self.agruparMedia().massaResidual(period=['01-01-1990', '01-01-2021'], basis = ['01-01-1990', '01-01-2010']).recorteTemporal()
        
        plot(vazao = vazao_media_verao, loess = loess, media = media, mediana = mediana, rippl = rippl)
