import xarray as xr
import rioxarray as rio
import geopandas as gpd
import geojson
import re
from libs import series
import pandas as pd
from shapely.geometry import mapping

def preparar_para_recorte(dataset, crs="epsg:4326", xdim="lon", ydim="lat"):
    """
    Prepara o dataset para o recorte.
    
    Args:
    dataset: xarray.DataArray
        Dataset pra ser recortado.
    """

    dataset = dataset.rio.set_spatial_dims(x_dim=xdim, y_dim=ydim) 
    dataset = dataset.rio.write_crs(crs)
    
    return dataset


def recorteGrade(shapefile, 
                 netcdf):
    """
    Recebe um objeto xarray.DataArray para ser recortado de acordo com o contorno do shapefile.
    
    Args:
    shapefile: str.
        Caminho para o arquivo shp.
        
    netcdf: xarray.DataArray.
        Dado em grade.
    
    Retorna:
    xarray.Dataset recortado dentro do contorno do shapefile.
    """

    dados_preparados = preparar_para_recorte(netcdf)
    dados_recortados = dados_preparados.rio.clip(shapefile["geometry"], crs=dados_preparados.rio.crs)

    return dados_recortados


def fit(caminho):
    """
    Inicializa o dataset.
    """
    dataset = xr.open_dataset(caminho)
   
    if 'expver' in dataset.coords:
        dataset = dataset.isel(expver=0)
        dataset = dataset.drop('expver')
    if 'time_bnds' in dataset:
        dataset = dataset.drop('time_bnds')

    # padroniza o nome do array de latitude e longitude
    # idem pro restante (lat, lon, olr, uwnd, vwnd)
    for variavel in dataset.variables:
        if variavel == "latitude":
            dataset = dataset.rename({variavel: 'lat'})
        elif variavel == "longitude":
            dataset = dataset.rename({variavel: 'lon'})
        elif variavel == "u":
            dataset = dataset.rename({variavel: 'uwnd'})
        elif variavel == "v":
            dataset = dataset.rename({variavel: 'vwnd'})
        elif variavel == "z":
            dataset = dataset.rename({variavel: 'hgt'})
        elif variavel == "mtnlwrf":
            dataset = dataset.rename({variavel: 'olr'})   

    return Grade(dataset)


class Grade:
    """
    Classe para o pr??-processamento de dados em grade.
        
    Atributos:
    dataset: xarray.Dataset
        Acessa o dataset
    """
    
    def __init__(self, dataset):
        self.dataset = dataset
        # assumindo um dataset de 1 vari??vel por vez :T depois vou melhorar isso
        self.variavel = [variaveis for variaveis in [*dataset] if 'bnds' not in variaveis][0]


    def atribuir(self, attrs_dict):
        """
        Armazena informa????es no atributo dataset do objeto Grade.
        
        Args:
        attrs_dict: dict
            Dicion??rio contendo os nomes dos atributos e seus valores.
        """
        for keys, values in attrs_dict.items():
            self.dataset.attrs[keys] = values
    
    
    def agruparMedia(self, freq="MS"):
        """
        Retorna a s??rie temporal ap??s passado o resample de acordo com a frequ??ncia escolhida. 
            
        Kwargs:
        freq: str. Default: "MS"
            Frequ??ncia de agrupamento.
        """
        dataset = self.dataset.resample(time=freq).mean()
        
        return dataset
    
    
    def mlt(self, freq, periodos):
        """
        Retorna a m??dia de longo termo do dataset.
        
        Args:
        freq: str. ["month", "year", "day", "season"]
            Frequ??ncia de agrupamento.
            
        periodos: str, list.
            String contendo a data de recorte ou uma lista de strings para um recorte de intervalo.
        """
        frequencia = f"time.{freq}"
        dataset = self.recorteIntervalo(periodos).groupby(frequencia).mean()
        
        return dataset


    def anomalia(self, freq, periodos, bases, serie = False):
        """
        Retorna um dataset de anomalias a partir da m??dia de dois per??odos distintos.
        
        Args:
        freq: str. ["month", "year", "day", "season"]
            Frequ??ncia de agrupamento.
            
        periodos: list.
            Lista de strings para um recorte de intervalo.
            
        bases: list.
            Lista de strings para a base do per??odo de anomalias.
            
        Kwargs:
        serie: bool.
            Caso True, o kwarg s??rie faz com que o m??todo retorne a s??rie de anomalias.
            O default retorna a anomalia da m??dia entre dois per??odos.
        """
        frequencia = f"time.{freq}"
        
        if serie == True:
            periodo = self.recorteIntervalo(periodos)
            
        elif serie == False:
            periodo = self.recorteIntervalo(periodos).groupby(frequencia).mean()
            
        base = self.mlt(freq = freq, periodos = bases)

        anomalias = periodo - base

        return anomalias


    def recorteAno(self, meses):
        """
        Retorna a s??rie temporal recortada no mesmo per??odo do ano, para todos os anos. 

        Args:
        meses: list.
        Lista de meses expressos em n??meros inteiros. 
        """
        if isinstance(meses, list):
            meses_selecionados = []
            
            for mes in meses:
                recorte = self.dataset.where(self.dataset['time.month'] == mes,
                                    drop=True)
                
                meses_selecionados.append(recorte)


            recorte_total = xr.concat(meses_selecionados, dim='time')
            dataset = recorte_total.sortby(recorte_total.time).dropna(dim='time')
            
        return Grade(dataset)


    def recorteIntervalo(self, periodos):
        """
        Recorta o intervalo da s??rie.

        Args:
        periodos: str, list.
            String contendo a data de recorte ou uma lista de strings para um recorte de intervalo.
        """
        if periodos is not None:

            if isinstance(periodos, list):
                dataset = self.dataset.sel(time=slice(periodos[0], periodos[1]))

            else:
                dataset = self.dataset.sel(time=periodos)

        return dataset


    def sazonalidade(self):
        """
        Retorna a sazonalidade anual.
        """
        # d?? pra melhorar isso aqui depois
        data_list = []
        time_list = []
        
        for i in range(0, len(self.dataset.time.values), 12):
            d_slice = self.dataset.isel(time=slice(i, i + 12))
            d_season = d_slice.groupby('time.season').mean()
            data_list.append(d_season)
            time_list.append(pd.to_datetime(self.dataset.time.values[i]).year)

        idx = pd.Index(time_list, name='time')
        ds = xr.concat(data_list, dim=idx)
        
        return ds
    
    
    def aave(self):
        """
        Retorna a m??dia regional.
        """
        dataset = self.dataset.mean(dim=('lat', 'lon'))

        return dataset
