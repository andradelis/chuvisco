import pandas as pd

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
    
    serie['datas'] = datas

    return serie


def mlt(serie):
    """
    Retorna a média de longo termo da série.
    
    Args:
    df: pd.Series.
        Série temporal.
        
    Retorna:
    pd.Series:
        Média de longo termo.
    """
    
    # obtendo a média de longo termo
    media_longo_termo = serie.groupby(by = serie.index.month).mean()
    
    return media_longo_termo


def recorteAno(serie, meses):
    """
    Retorna a série temporal recortada no mesmo período do ano, para todos os anos. 
    Uma série temporal não linear assume valores categóricos para o vetor de data.

    Args:
    serie: pd.Series.
        Série temporal.
    
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
