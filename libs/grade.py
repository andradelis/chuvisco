import xarray as xr
import pandas as pd

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
    Classe para o pré-processamento de dados em grade.
        
    Atributos:
    dataset: xarray.Dataset
        Acessa o dataset
    """
    
    def __init__(self, dataset):
        self.dataset = dataset
        # assumindo um dataset de 1 variável por vez :T depois vou melhorar isso
        self.variavel = [variaveis for variaveis in [*dataset] if 'bnds' not in variaveis][0]


    def atribuir(self, attrs_dict):
        """
        Armazena informações no atributo dataset do objeto Grade.
        
        Args:
        attrs_dict: dict
            Dicionário contendo os nomes dos atributos e seus valores.
        """
        for keys, values in attrs_dict.items():
            self.dataset.attrs[keys] = values
    
    
    def agruparMedia(self, freq="MS"):
        """
        Retorna a série temporal após passado o resample de acordo com a frequência escolhida. 
            
        Kwargs:
        freq: str. Default: "MS"
            Frequência de agrupamento.
        """
        dataset = self.dataset.resample(time=freq).mean()
        
        return dataset
    
    
    def mlt(self, freq, periodos):
        """
        Retorna a média de longo termo do dataset.
        
        Args:
        freq: str. ["month", "year", "day", "season"]
            Frequência de agrupamento.
            
        periodos: str, list.
            String contendo a data de recorte ou uma lista de strings para um recorte de intervalo.
        """
        frequencia = f"time.{freq}"
        dataset = self.recorteIntervalo(periodos).groupby(frequencia).mean()
        
        return dataset


    def anomalia(self, freq, periodos, bases, serie = False):
        """
        Retorna um dataset de anomalias a partir da média de dois períodos distintos.
        
        Args:
        freq: str. ["month", "year", "day", "season"]
            Frequência de agrupamento.
            
        periodos: list.
            Lista de strings para um recorte de intervalo.
            
        bases: list.
            Lista de strings para a base do período de anomalias.
            
        Kwargs:
        serie: bool.
            Caso True, o kwarg série faz com que o método retorne a série de anomalias.
            O default retorna a anomalia da média entre dois períodos.
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
        Retorna a série temporal recortada no mesmo período do ano, para todos os anos. 

        Args:
        meses: list.
        Lista de meses expressos em números inteiros. 
        """
        if isinstance(meses, list):
            meses_selecionados = []
            
            for mes in meses:
                recorte = self.dataset.where(self.dataset['time.month'] == mes,
                                    drop=True)
                
                meses_selecionados.append(recorte)


            recorte_total = xr.concat(meses_selecionados, dim='time')
            dataset = recorte_total.sortby(recorte_total.time).dropna(dim='time')
            
        return dataset


    def recorteIntervalo(self, periodos):
        """
        Recorta o intervalo da série.

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
        # dá pra melhorar isso aqui depois
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
        Retorna a média regional.
        """
        dataset = self.dataset.mean(dim=('lat', 'lon'))

        return dataset
