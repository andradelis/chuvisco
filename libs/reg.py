import numpy as np
import xarray as xr


def tendencia(d, deg=1, norm=True, xdim="lon", ydim="lat"):

    # função para computar a tendência linear da série temporal
    def _regredir(x, deg=deg):
        pf = np.polyfit(x.time, x.values, deg=deg)
        da = xr.DataArray(pf[0])
        return da
    
    def _normalizar(d):
        standardized = xr.apply_ufunc(
            lambda x, m, s: (x - m) / s,
            d,
            d.mean(),
            d.std(),
        )
        return standardized
    
    dados = d.copy()
    dados['time'] = dados["time"].astype("float64")

    # estaca a lat e lon em uma dimensão única 
    stacked = dados.stack(allpoints=[ydim, xdim])

    # aplica a função de tendência à dimensão unificada
    tend = stacked.groupby('allpoints').apply(_regredir)
    
    tend_unstack = tend.unstack('allpoints')
    
    saida = _normalizar(tend_unstack) if norm else tend_unstack
    
    return saida