from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist 
from libs import series
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL


def massaresidual(serie, 
                  periodo =  ['1979-01-01', '2021-12-01'], 
                  base = ['1979-01-01', '2010-01-01']):
    """
    Retorna a massa residual do dataframe.
    
    Args:
    serie: pd.Series.
        Dados.
        
    Kwargs:
    periodo: list. Default: ['1979-01-01', '2018-12-01']
        Intervalo de tempo da série de anomalia.
        
    base: list. Default: ['1979-01-01', '2010-01-01']
        Intervalo base.
    """
    
    anom = series.anomalia(serie)
    
    # calculando a série de massa residual
    rippl_rebatido = anom['anomalia']/anom['mlt']
    
    # soma cumulativa da série de massa residual
    massa_residual = np.cumsum(rippl_rebatido)
    
    massa_residual.index = anom['datas']
    
    return massa_residual.to_frame(name = f"residual_{serie.name}")


def mediaMovel(df, janela):
    """
    Retorna a média móvel da série temporal.

    Args:
    janela: int
        Janela temporal da média móvel.
    """

    saida = df.rolling(janela).mean()

    return saida


def medianaMovel(df, janela):
    """
    Retorna a mediana móvel da série temporal.

    Args:
    janela: int
        Janela temporal da mediana móvel.
    """

    saida = df.rolling(janela).median()

    return saida
    

def loess(df,
          tipo,
          **loess_kw):
    """
    Regressão local LOESS.

    Args:
    df: pd.DataFrame. 
        Dados.
        
    tipo: {'tendência', 'sazonalidade', 'observado', 'residual'}
        Tipo de análise.
    
    Kwargs:
    **loess_kw: statsmodels.tsa.seasonal.STL kwargs.
    """

    # decomposição de loess
    stl = STL(df,
              **loess_kw)
    
    res = stl.fit()

    # retorna resultado de acordo com o tipo de análise
    if tipo == 'tendência':
        res = res.trend
    elif tipo == 'sazonalidade':
        res = res.seasonal
    elif tipo == 'observado':
        res = res.observed
    elif tipo == 'residual':
        res = res.resid

    return res
    

def standard(df, 
             **scaler_kw):
    """
    Normaliza um dataframe.
    
    Args:
    df: pd.DataFrame.
    
    Kwargs:
    **scaler_kw: sklearn.preprocessing.StandardScaler kwargs.
    
    Retorna:
    pd.Dataframe: Dataframe normalizado.
    """
    
    df_final = StandardScaler(**scaler_kw).fit_transform(df)

    return df_final


def curvatura(K, 
              y,
              **knee_kw):
    """
    Busca o ponto de maior curvatura da curva.
    
    Args:
    K: array
        Eixo x.
    
    y: array
        Eixo y.
    
    Kwargs:
    knee_kwargs: kneed.KneeLocator kwargs.
    
    Retorna:
    int: Ponto de maior curvatura.
    """
    
    kn = KneeLocator(K,
                     y,
                     **knee_kw).knee
    
    return kn


def cotovelo(df,
             metodo = 'inertia'):
    """Performa o método do cotovelo para obter o número ótimo de clusters em um dataframe.
    
    Args:
    df: pd.DataFrame.
        Dados.
        
    Kwargs:
    metodo: str. {'distortion', 'inertia'}
        Método para a determinação de clusters.
        
    Retorna:
    int: Número de clusters
    """
    
    y = {}  
    K = range(1, len(df.columns) + 1)

    for k in K:   
        kmeanModel = KMeans(n_clusters=k).fit(df)
        
        if metodo == 'distortion':
            y.update({k: sum(np.min(cdist(df,
                                        kmeanModel.cluster_centers_, 
                                        'euclidean'),
                                        axis=1)) / df.shape[0]}) 
            
        elif metodo == 'inertia':
            y.update({k: kmeanModel.inertia_}) 

    knee_kwargs = {'curve': 'convex', 'direction': 'decreasing'}
    n_clusters = curvatura(K, [*y.values()], **knee_kwargs)
        
    return n_clusters


def mannkendall(df, 
                ref = "hamed rao",
                retorna_df = False, 
                **mk_kw):
    """
    Retorna o teste de Mann-Kendall para a distribuição dos eventos na série histórica.

    Args:
    df: pd.DataFrame 1D.
        Série.

    Kwargs:
    ref: str. Default: "hamed rao"
        Referência para o teste de Mann-Kendall. ['hamed rao', 'yue wang', 'trend-free prewhitening', 'prewhitening', 'seasonal']

    retorna_df: bool. Default: False
        Caso verdadeiro, retorna um pd.DataFrame com os atributos do teste de Mann-Kendall.

    mann_kendall_sazonal: int. Default: 12
        Período para o teste sazonal de Mann-Kendall.
        
    Retorna:
    pymannkendall ou pd.DataFrame
    """

    if ref == "hamed rao":
        resultado = mk.hamed_rao_modification_test(df, **mk_kw)

    elif ref == "yue wang":
        resultado = mk.yue_wang_modification_test(df, **mk_kw)

    elif ref == "trend-free prewhitening":
        resultado = mk.trend_free_pre_whitening_modification_test(df, **mk_kw)

    elif ref == "prewhitening":
        resultado = mk.pre_whitening_modification_test(df, **mk_kw)

    elif ref == 'seasonal':
        resultado = mk.seasonal_test(df, **mk_kw)

    if retorna_df == True:
        trend, h, p, z, Tau, s, var_s, slope, intercept = resultado
        mannKendall_dict = {df.columns[0]: {"Tendência": trend,
                                            "Valor p": p,
                                            "Tau": Tau,
                                            "Z": z,
                                            "Sen": slope}}

        return pd.DataFrame.from_dict(mannKendall_dict)

    elif retorna_df == False:
        
        return resultado
    

def kmeans(df,
            n_clusters = 'cotovelo', 
            metodo = 'inertia', 
            normalizar = False):
    """
    Separa um dataframe normalizado em clusters.
    
    Args:
    df: pd.DataFrame.
        Dataframe a ser clusterizado.
        
    Kwargs:
    n_clusters: 'cotovelo' ou inteiro. Default: 'cotovelo'
        Número de clusters para o agrupamento. Se n_clusters = 'cotovelo', utiliza o método do cotovelo para encontrar o número ótimo de clusters.
        
    normalizar: bool. Default: False
        Normalizar o dado.
        
    metodo: {'distortion', 'inertia'}. Default: 'distortion'
        Método para a determinação de clusters. Funciona se cotovelo = True.
        
    Retorna:
    pd.DataFrame: Linha de clusters concatenada ao dataframe original.
    """
    
    if normalizar == True:
        df = standard(df)
    
    if n_clusters == "cotovelo":
        n_clusters = cotovelo(df, metodo)
    
    kmeanModel = KMeans(n_clusters=n_clusters)
    
    kmeans_ = kmeanModel.fit_predict(df.T)
    df_kmeans = pd.DataFrame(kmeans_, index = df.T.index, columns = ["cluster"])
    
    df_final = pd.concat([df_kmeans, df.T], axis = 1).T
    
    return df_final


def cluster(df, 
            **linkage_kw):
    """
    Clusterização hierárquica de um dataframe.
    
    Args:
    df: pd.DataFrame.
        Dados.

    Kwargs:
    linkage_kw: scipy.cluster.hierarchy.cluster kwargs.
    
    Retorna:
    Matriz de agrupamento da clusterização.
    """
    
    df = df.T
    linked = linkage(df, **linkage_kw)
    
    return linked


def dendrograma(matriz, 
                **dendro_kw):
    """
    Retorna o dendrograma da matriz de agrupamento.
    
    Args:
    matriz: array
        Matriz de agrupamento.
        
    Kwargs:
    dendro_kw: scipy.cluster.hierarchy.dendrogram
    
    Retorna:
    matplotlib.pyplot.figure, matplotlib.pyplot.axes
    """    
    # criando a imagem
    fig, ax = plt.subplots(figsize=(13,5))
    
    # plotando o dendrograma
    dendrogram(matriz,
                orientation = dendro_kw.get('orientation', 'top'),
                distance_sort = dendro_kw.get('distance_sort', 'descending'),
                ax = ax, 
                above_threshold_color = "#3E4B4B", 
                **dendro_kw)

    # configurações do ax
    ax.axhline(y = 175, linewidth=3, linestyle = "dashed", color='lightgray')
    ax.set_ylabel("Distância euclidiana", fontsize = 14)

    right, top, bottom = ax.spines["right"], ax.spines["top"], ax.spines["bottom"]
    right.set_visible(False)
    top.set_visible(False)
    bottom.set_visible(False)

    return fig, ax


def pca(df, 
        **pca_kw):
    """
    Retorna as componentes principais de um dataframe. Caso um número de componentes n_components não seja passado, utiliza o número de colunas do dataframe.
    
    Args:
    df: pd.DataFrame.
        Dados.
    
    Kwargs:
    **pca_kw: sklearn.decomposition.PCA kwargs
    
    Retorna: 
    lista contendo [df_de_PCs, {sklearn.decomposition.PCA attributes}]
    """

    dfs = []  
    
    n_components = pca_kw.get('n_components', len(df.columns))
    
    inicializacao = PCA(n_components = n_components)
    pca_ = inicializacao.fit_transform(df)

    for i in range(n_components):
        dfs.append(pd.DataFrame(pca_[:,i],
                                index = df.index,
                                columns = [f"PC{i + 1}"]))
        
    df_final = pd.concat(dfs, axis = 1)
    
    return [df_final,
            {"Eixo principal por feature": inicializacao.components_, 
            "Variância": inicializacao.explained_variance_ratio_, 
            "Média por feature": inicializacao.mean_, 
            "Número estimado de componentes": inicializacao.n_components_,
            "Variância residual": inicializacao.noise_variance_}]
    

