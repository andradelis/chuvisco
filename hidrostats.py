import climsy
import os
import matplotlib.pyplot as plt
import pandas as pd
import pymannkendall as mk
import xarray as xr
import datetime as dt
import numpy as np
import matplotlib.patches as mpatches
from dateutil.relativedelta import relativedelta



def csv2ds(caminho, **kwargs):
    """
    Lê um arquivo .csv e retorna como um objeto xarray.Dataset.

    Parâmetros:
    caminho: str
        Diretório do arquivo .csv.

    Kwargs:
    format: str
        Formato de data no índice da série. Default: ['%d/%m/%Y'].
    """

    csv = pd.read_csv(caminho)
    csv.index = pd.to_datetime(csv.index, format=kwargs.get('format', '%d/%m/%Y'))

    ds = xr.Dataset.from_dataframe(csv)

    return ds


class Vazao():

    def __init__(self, caminho_ou_dataset, **kwargs):
        """
        Classe de análise de vazões.

        Parâmetros:
        caminho_ou_dataset: str, pandas.DataFrame, xarray.Dataset ou xarray.DataArray.
            Diretório do arquivo .csv ou um objeto xarray.Dataset. Caso seja passado um objeto da classe xarray.DataArray ou pandas.DataFrame, o objeto será transformado em dataset.

        Kwargs:
        format: str
            Formato de data no índice da série. Default: ['%d/%m/%Y'].
        """

        if isinstance(caminho_ou_dataset, str):
            vazao = csv2ds(caminho = caminho_ou_dataset, **kwargs)
        elif isinstance(caminho_ou_dataset, xr.Dataset):
            vazao = caminho_ou_dataset
        elif isinstance(caminho_ou_dataset, xr.DataArray):
            vazao = caminho_ou_dataset.to_dataset()
        elif isinstance(caminho_ou_dataset, pd.DataFrame):
            vazao = xr.Dataset.from_dataframe(caminho_ou_dataset)

        self.vazao = vazao
        self.usina = str(vazao.to_array().isel(variable=0)['variable'].values)
        self.dir_saida = os.path.join(os.getcwd(), "saída")


    def agruparMedia(self, freq="MS"):
        """
        Retorna a série temporal após passado o resample de acordo com a frequência escolhida. A sintaxe de frequẽncia segue o padrão do Pandas.

        Kwargs:
        freq: str. Default: "MS"
            Frequência de agrupamento.
        """

        vazao = self.vazao

        return Vazao(vazao.resample(time = freq).mean())


    def indiceCategorico(self, array, step = relativedelta(months=1)):
        """
        Converte o vetor temporal para categórico.

        Parâmetros:
        array: xr.DataArray
            Vetor temporal.

        Kwargs:
        step: dateutil.relativedelta
            Diferença esperada entre as datas.
        """

        def diasConsecutivos(a, b, step = relativedelta(months=1)):
            """
            Checa se duas datas estão separadas por um período de tempo.

            Parâmetros:
            a: datetime64[ns]
                Primeira data.

            b: datetime64[ns]
                Segunda data.

            Kwargs:
            step: dateutil.relativedelta
                Diferença esperada entre as datas.
            """

            return (a + step) == b


        # se a função diasConsecutivos ao longo de todo o vetor for verdadeira, o algoritmo retorna o vetor de tempo inalterado. Senão, transforma o objeto de data em string.
        if all(diasConsecutivos(pd.to_datetime(array[i].values), pd.to_datetime(array[i+1].values, step)) for i in range(len(array) - 1)):
            pass
        else:
            array["time"] = array["time"].dt.strftime("%m-%Y")

            return array["time"]


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
        Retorna o diagrama de massa residual (diagrama de Rippl rebatido) segundo a equação:

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
        y = self.vazao[self.usina]
        # Obter n valores entre 0 e 1, onde n é igual ao comprimento do vetor contendo os valores de vazão natural
        x = np.linspace(0, 1, len(self.vazao[self.usina]))

        # Obter os coeficientes para o ajuste de um polinômio p(x) = p_0*x^n + p_1*x^(n-1) + ... + p_n, onde n é o grau do polinômio.
        coefs = np.polyfit(x, y, grau_polinomio)
        # Os coeficientes obtidos são parâmetros para a instância da classe de operações polinomiais do numpy
        f = np.poly1d(coefs)
        polinomio = f(x)

        return polinomio


    def mannKendall(self, ref = "hamed rao", retorna_df = False, **kwargs):
        """
        Retorna o teste de Mann-Kendall para a distribuição dos eventos na série histórica.

        Kwargs:
        ref: str. Default: "hamed rao"
            Referência para o teste de Mann-Kendall. ['hamed rao', 'yue wang', 'trend-free prewhitening', 'prewhitening', 'seasonal']

        retorna_df: bool. Default: False
            Caso verdadeiro, retorna um pd.DataFrame com os atributos do teste de Mann-Kendall.

        mann_kendall_sazonal: int. Default: 12
            Período para o teste sazonal de Mann-Kendall.
        """

        vazao = self.vazao

        y = vazao[self.usina]

        if ref == "hamed rao":
            resultado = mk.hamed_rao_modification_test(y)

        elif ref == "yue wang":
            resultado = mk.yue_wang_modification_test(y)

        elif ref == "trend-free prewhitening":
            resultado = mk.trend_free_pre_whitening_modification_test(y)

        elif ref == "prewhitening":
            resultado = mk.pre_whitening_modification_test(y)

        elif ref == 'seasonal':
            resultado = mk.seasonal_test(y, period=kwargs.get('mann_kendall_sazonal', 12))

        if retorna_df == True:
            trend, h, p, z, Tau, s, var_s, slope, intercept = resultado
            mannKendall_dict = {self.usina: {"Tendência": trend,
                                        "Valor p": p,
                                        "Tau": Tau,
                                        "Z": z,
                                        "Sen": slope,
                                        "H": h,
                                        "Var(S)": var_s,
                                        "Intercept": intercept,
                                        "S": s}}

            return pd.DataFrame.from_dict(mannKendall_dict)


        return resultado


    def loess(self, **kwargs):
        """
        Retorna uma curva calculada a partir da regressão local da série temporal.

        Kwargs:
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


    def plot(self, ax, dado, **kwargs):
        """
        Configuração padrão do gráfico.

        Parâmetros:
        ax: matplotlib.axes._subplots.AxesSubplot

        dado: xarray.DataArray
            Array da variável

        Kwargs:
        fontsize: int. Default: 13
            Tamanho da fonte nos eixos x e y.

        label: str. Default: ''
            Label vinculada à série, a ser mostrado na legenda.

        passo: int. Default: 16
            Número de entradas no eixo x.

        color: str. Default: '#00AF91'
            Cor do gráfico.

        linewidth: float. Default: 1
            Espessura da linha do gráfico.

        alpha: float. Default: 1
            Transparência da linha do gráfico.
        """

        # removendo as spines indesejadas
        right = ax.spines["right"]
        right.set_visible(False)
        up = ax.spines["top"]
        up.set_visible(False)

        # checando o tipo do índice para que seja alterado para string caso não seja contínuo, garantindo que o dado não tenha lacunas
        dado["time"] = self.indiceCategorico(dado["time"])
        ax.set_ylabel(kwargs.get('set_ylabel', 'Vazão natural\n(m³/s)'), fontsize = kwargs.get('fontsize', 15))
        ax.plot(dado['time'], dado, label = kwargs.get('label', ''), color = kwargs.get('color', '#00AF91'), linewidth=kwargs.get('linewidth', 1), alpha=kwargs.get('alpha', 1))

        # caso a variável independente "time" não seja do tipo data, formata o eixo de data
        if dado["time"].dtype == 'object':
            start, end = ax.get_xlim()
            ax.xaxis.set_ticks(np.linspace(start, end, kwargs.get('passo', 17)))
            plt.setp(ax.get_xmajorticklabels(), fontsize = kwargs.get('fontsize', 15))
            plt.setp(ax.get_ymajorticklabels(), fontsize = kwargs.get('fontsize', 15))


    def analiseVazao(self, recorte = "", janela_movel = (10*6), freq = "MS", mann_kendall_ref = "hamed rao", recorte_temporal = "rainy season", grau = 4, mann_kendall_sazonal = 12, exportar = True, type = 'trend', seasonality = 7, periodo_loess = 6, robust = False, seasonal_deg = 1, trend_deg = 1, low_pass_deg = 1, **kwargs):
        """
        Compilado de métodos da classe Vazao que facilita a visualização da tendência nas séries históricas de vazão.

        Kwargs:
        recorte: str. Default: ""
            Recorte temporal da série. Ex: Se a série foi recortada para o período úmido, pode ser específicado recorte = " de Outubro a Março".

        janela_movel: int. Default: 60
            Janela temporal para o cálculo da média e mediana móvel.

        freq: str. Default: "MS"
            Frequência da série.

        mann_kendall_ref: str. Default: "hamed rao"
            Metodologia para o teste de Mann-Kendall.

        recorte_temporal: str. Default: "rainy season"
            Argumento passado para a função de recorte temporal.

        grau: int. Default: 4
            Grau do polinômio de ajuste.

        mann_kendall_sazonal: int. Default: 12
            Período para o teste sazonal de Mann-Kendall.

        exportar: bool. Default: True
            Exporta a imagem para o diretório de saída.

        type: 'trend', 'seasonality', 'observed', 'residual'. Default: 'trend'
            Tipos de análise STL.

        seasonality: int > 3
            Fator de suavização da curva. Ver statsmodels.tsa.seasonal.STL.

        periodo_loess: int
            Ver statsmodels.tsa.seasonal.STL.

        robust: bool
            Ver statsmodels.tsa.seasonal.STL.

        seasonal_deg: int
            Ver statsmodels.tsa.seasonal.STL.

        trend_deg: int
            Ver statsmodels.tsa.seasonal.STL.

        low_pass_deg: int
            Ver statsmodels.tsa.seasonal.STL.

        dir_saida: str. Default: /saída/img/
            Nome do diretório de saída.
        """

        def analisePlot(vazao, loess, media, mediana, rippl, mann_kendall, recorte, grau):
            """
            Plotagem fácil da análise de vazões.

            Parâmetros:
            vazao: xarray.DataArray
                Array contendo as vazões.

            loess: xarray.DataArray
                Array contendo a decomposição de LOESS.

            media: xarray.DataArray
                Array contendo a média móvel.

            mediana: xarray.DataArray
                Array contendo a mediana móvel.

            rippl: hidrostats.Vazao
                Objeto Vazao contendo os valores da curva de massa residual.

            mann_kendall: list
                Saída do teste de Mann-Kendall contendo a tendência, a hipótese, o valor p, z, Tau, s, a variância S, a declividade de Sen e o intercept. (Ver pymannkendall.)

            recorte: str
                Recorte temporal da série. Ex: Se a série foi recortada para o período úmido, pode ser específicado recorte = " de Outubro a Março".

            grau: int
                Grau do polinômio de ajuste.
            """

            fig, ax = plt.subplots(nrows=3, figsize = (24, 16))

            # Plot da série de vazão e a regressão local do STL
            self.plot(ax[0], dado = vazao.vazao[self.usina], color = '#00AF91', linewidth = 2, alpha = 0.8)
            vazao_patch = mpatches.Patch(color='#00AF91', label="Vazão natural (m³/s)")
            self.plot(ax[0], dado = loess[self.usina], color = "#3E4B4B")
            loess_patch = mpatches.Patch(color='#3E4B4B', label = f"Regressão local ({type})")

            ax[0].legend(fancybox=True, handles = [vazao_patch, loess_patch], loc="upper left", fontsize = 14, borderpad=0.5)

            # Plot da série de média e mediana móveis
            self.plot(ax[1], dado = media[self.usina], color = '#00AF91', linewidth = 2, alpha = 0.8)
            media_patch = mpatches.Patch(color='#00AF91', label = f"Média móvel ({janela_movel} meses)")
            self.plot(ax[1], dado = mediana[self.usina], label = f"Mediana móvel ({janela_movel} meses)", color = "#3E4B4B", linewidth=1.1)
            mediana_patch = mpatches.Patch(color='#3E4B4B', label = f"Mediana móvel ({janela_movel} meses)")

            ax[1].legend(fancybox=True, handles = [media_patch, mediana_patch], loc="upper left", fontsize = 14, borderpad=0.5)

            # Plot da série de Rippl rebatido e a aproximação polinomial de ajuste de curva
            self.plot(ax[2], dado = rippl.vazao[self.usina],  color = '#00AF91', linewidth = 2, alpha = 0.8)
            rippl_patch = mpatches.Patch(color='#00AF91', label = "Anomalia acumulada")
            ax[2].plot(rippl.regPoli(grau), color = "#3E4B4B", linestyle='--')
            mmq_patch = mpatches.Patch(color='#3E4B4B', label = "M.M.Q")

            ax[2].legend(fancybox=True, handles = [rippl_patch, mmq_patch], loc="upper left",fontsize = 14)

            # Títulos dos axes
            ax[0].set_title(f'Vazão natural afluente em {self.usina}\nMédia mensal{recorte}\n\n', fontsize = 18)
            ax[1].set_title(f'Média e mediana móvel para a vazão de {self.usina} mensal{recorte}\n\n', fontsize = 18)
            ax[2].set_title(f'\nDiagrama de Rippl a partir das anomalias de vazão natural afluente no reservatório de {self.usina}\nMédia mensal{recorte} em relação à média de longo termo entre {pd.to_datetime(loess["time"][0].values).year} e 2010\n\n', fontsize = 18)

            # setando o segundo ax com os mesmos xticks que o terceiro
            ax[1].set_xticks(ax[2].get_xticks())

            # Informações de Mann-Kendall no ax
            trend, h, p, z, Tau, s, var_s, slope, intercept = mann_kendall
            ax[0].annotate(f"Mann-Kendall ({mann_kendall_ref})\nTendência: {trend}\np: {round(p, 8)}\nZ: {round(z, 8)}\nTau: {round(Tau, 8)}\nSen: {round(slope, 8)}",
                            horizontalalignment='right', xy=(0.99, 0.75), xycoords="axes fraction", size = 16, bbox=dict(boxstyle="round", alpha=0.25, facecolor = "white", edgecolor = "grey"))

            fig.tight_layout()

            if exportar:
                self.exportarImagem(dir_saida = kwargs.get('dir_saida', os.path.join(self.dir_saida, "img", self.usina)))

            return fig, ax

        # agrupa a vazão em médias de acordo com a frequência freq e recorta a série de acordo com o parâmetro series
        vazao_media_verao = self.agruparMedia(freq = freq).recorteTemporal(series = recorte_temporal)

        # performa o teste de Mann-Kendall de acordo com o parâmetro ref, que dá o teste desejado, e o período de sazonalidade mann_kendall_sazonal, caso exista
        mk = vazao_media_verao.mannKendall(ref = mann_kendall_ref, mann_kendall_sazonal = mann_kendall_sazonal)

        # performa a decomposição de loess de acordo com os parâmetros (Ver hidrostats.Vazao.loess)
        loess = vazao_media_verao.loess(type = type, seasonality = seasonality, period = periodo_loess, robust = robust, seasonal_deg = seasonal_deg, trend_deg = trend_deg, low_pass_deg = low_pass_deg)

        # média e mediana móveis de acordo com a janela temporal
        media = vazao_media_verao.mediaMovel(janela = janela_movel)
        mediana = vazao_media_verao.medianaMovel(janela = janela_movel)

        # diagrama de Rippl rebatido para a análise de anomalias acumuladas
        rippl = self.agruparMedia(freq = freq).massaResidual(period=[vazao_media_verao.vazao["time"][0], vazao_media_verao.vazao["time"][-1]], basis = [vazao_media_verao.vazao["time"][0], '12-31-2010']).recorteTemporal(series = recorte_temporal)

        # plot fácil para o compilado de análises
        analisePlot(vazao = vazao_media_verao, loess = loess, media = media, mediana = mediana, rippl = rippl, mann_kendall = mk, recorte = recorte, grau = grau)


    def exportarImagem(self, **kwargs):
        """
        Exportar o arquivo de imagem.

        Kwargs:
        dir_saida: str. Default: hidrostats/saída/img/
            Nome do diretório de saída.

        arquivo: str. Default: xarray.DataArray.name
            Nome do arquivo a ser exportado.

        transparent: bool. Default: False
            Transparência da imagem.
        """

        diretorio = os.path.join(self.dir_saida, "img", self.usina)

        # criando os diretórios de imagem
        try:
            os.makedirs(kwargs.get('dir_saida', diretorio))
        except FileExistsError:
            pass

        plt.savefig(f"{os.path.join(kwargs.get('dir_saida', diretorio), kwargs.get('arquivo', self.usina))}.png", transparent = kwargs.get('transparent', False))
