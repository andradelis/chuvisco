import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def base(title, lat, lon, figsize=(18, 8), proj=ccrs.PlateCarree()):
    
    def _geoaxes_format(ax):
        gl = ax.gridlines(crs=ccrs.PlateCarree(),
                          draw_labels=True,
                          linewidth=1,
                          color='black',
                          alpha=0.3,
                          linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_left = True
        gl.ylabels_right = False
        gl.ylines = True
        gl.xlines = True
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 200, 40))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 10}

    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw=dict(projection=proj))

    _geoaxes_format(ax)
    ax.set_title(title, fontsize=18, y=1.03)
    ax.coastlines("50m", color = "#3E4B4B")
    ax.add_feature(cfeature.BORDERS, edgecolor = '#3E4B4B', linestyle='-', alpha=1, linewidth = 1)

    if lat is not None and lon is not None:
        ax.set_extent([lon[0], lon[1], lat[0], lat[1]])

    return fig, ax


def rapido(dados, recorte, contour_kws):
    fig, ax = base("", **recorte)
    dados.plot.contourf(**contour_kws, ax=ax, )
    return fig, ax
