import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta


def indiceCategorico(array, step = relativedelta(months = 1)):
    """
    Converte o vetor temporal em categórico.

    Args:
    array: pd.Series ou pd.DataFrame.
        Série temporal ou dataframe de séries temporais.
        
    Kwargs:
    step: dateutil.relativedelta
        Diferença esperada entre as datas.
    """

    def diasConsecutivos(a, b, step):
        """
        Checa se duas datas estão separadas por um período de tempo.

        Args:
        a: datetime64[ns]
            Primeira data.

        b: datetime64[ns]
            Segunda data.
        
        step: dateutil.relativedelta
            Diferença esperada entre as datas.
        """
        return (a + step) == b
    

    if all(diasConsecutivos(array.index[i], array.index[i+1], step = step) for i in range(len(array) - 1)):
        pass
    else:
        array.index = array.index.strftime("%m-%Y")

        return array


def regressao(serie, grau_polinomio):
    """
    Retorna um array contendo os valores para a curva de ajuste para a tendência da série segundo o método dos mínimos quadrados.

    Args:
    serie: pd.Series.
        Série temporal.
        
    grau_polinomio: int
        Grau do polinômio de ajuste.
    """

    # Obter o vetor de vazão natural
    y = serie
    # Obter n valores entre 0 e 1, onde n é igual ao comprimento do vetor contendo os valores de vazão natural
    x = np.linspace(0, 1, len(serie))

    # Obter os coeficientes para o ajuste de um polinômio p(x) = p_0*x^n + p_1*x^(n-1) + ... + p_n, onde n é o grau do polinômio.
    coefs = np.polyfit(x, y, grau_polinomio)
    # Os coeficientes obtidos são parâmetros para a instância da classe de operações polinomiais do numpy
    f = np.poly1d(coefs)
    polinomio = f(x)

    return polinomio


def anomalia(serie,
             periodo = ['1979-01-01',
                      '2018-12-01'], 
             base = ['1979-01-01',
                     '2010-01-01']):
    """
    Retorna a anomalia da série temporal.
    
    Args:
    serie: pd.Series.
        Série temporal.
        
    Kwargs:
    periodo: list. Default: ['1979-01-01', '2018-12-01']
        Intervalo da análise.

    base: list. Default: ['1979-01-01', '2010-01-01']
        Intervalo da média de longo termo.
        
    Retorna:
    pd.DataFrame:
        Dataframe contendo a coluna com o valor original, a coluna 'mlt' com o valor da média de longo termo e a coluna 'anomalia' com o valor da anomalia.

    """
    
    # obtendo o período de análise total
    # o fim do período precisa ter data menor ou igual à data limite fornecida
    fim = serie[serie.index<=periodo[1]]
    # o início do período precisa ter a data maior do que a data de início fornecida
    periodo = fim[fim.index>periodo[0]]

    # mesmo procedimento para o período de base
    fim_base = serie[serie.index<=base[1]]
    periodo_base = fim_base[fim_base.index>base[0]]

    # obtendo a média de longo termo
    media_longo_termo = mlt(periodo_base)
    
    # armazenando o valor de data original
    datas = periodo.index
    
    # transformando o índice de data pra mês para poder comparar os índices da série com a média de longo termo
    periodo.index = periodo.index.month
    
    # criando um dataframe da série no período solicitado
    serie = pd.DataFrame(periodo)
    
    # criando uma coluna só com os valores da media de longo termo
    serie['mlt'] = media_longo_termo

    serie['anomalia'] = (serie.iloc[:, 0] - serie['mlt'])
    
    serie.index = datas
    
    return serie


def mlt(serie):
    """
    Retorna a média de longo termo da série.
    
    Args:
    df: pd.Series ou pd.DataFrame.
        Série temporal.
        
    Retorna:
    pd.Series:
        Média de longo termo.
    """
    
    return serie.groupby(by = serie.index.month).mean()


def recorteAno(serie, meses):
    """
    Retorna a série temporal recortada no mesmo período do ano, para todos os anos. 

    Args:
    serie: pd.Series ou pd.DataFrame.
        Série temporal ou dataframe de séries temporais.
    
    meses: list.
        Lista de meses expressos em números inteiros. 
    
    Retorna:
    pd.Series:
        Série temporal recortada.  
    """
    
    if isinstance(meses, list):
        meses_selecionados = []

        for mes in meses:
            recorte = serie[serie.index.month==mes]
            meses_selecionados.append(recorte)

        df_recortado = pd.concat(meses_selecionados, axis = 0)
        df_recortado.sort_index(inplace=True)

    return df_recortado


def recorteIntervalo(df, periodos):
    """
    Recorta o intervalo da série.

    Args:
    df: pd.Series ou pd.DataFrame
        Série temporal ou dataframe de séries temporais.
        
    periodos: list.
        Lista de strings para um recorte de intervalo.
    """

    if isinstance(periodos, list):
        fim = df[df.index<=periodos[1]]
        df_recortado = fim[fim.index>=periodos[0]]
        
    return df_recortado


def agruparMedia(serie, freq="MS"):
    """
    Retorna a série temporal após passado o resample de acordo com a frequência escolhida. 
    
    Args:
    df: pd.Series ou pd.DataFrame.
        Série temporal ou dataframe de séries temporais.
        
    Kwargs:
    freq: str. Default: "MS"
        Frequência de agrupamento.
    """

    return serie.resample(freq).mean()


