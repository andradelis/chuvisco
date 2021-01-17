from libs import grade 
import xarray as xr
import numpy as np
import pandas as pd
import os
import calendar
import metpy.calc as mpcalc
from decimal import Decimal
import datetime

class Atmosfera(grade.Grade):
    
    def __init__(self, grade):
        self.dataset = grade.dataset
        self.variavel = grade.variavel

        
    def magnitude(self, grade):
        """
        Retorna a magnitude do vento.

        Args:
        grade: libs.grade.Grade
            Objeto libs.grade.Grade contendo um dataset com valores de vento na direção u ou v, complementar ao objeto Atmosfera.
        """
        dataset = self.dataset
        var = self.variavel

        dataset_2 = grade.dataset
        var_2 = grade.variavel

        if var in ['uwnd', 'vwnd'] and var_2 in ['uwnd', 'vwnd'] and var != var_2:
            dataset = ((dataset[var])**2 + (dataset_2[var_2])**2)**0.5
            magnitude = dataset.to_dataset(name='magnitude')

        return magnitude


    def divergencia(self, grade):
        """
        Retorna a divergência do vento.

        Args:
        grade: libs.grade.Grade
            Objeto libs.grade.Grade contendo um dataset com valores de vento na direção u ou v, complementar ao objeto Atmosfera.
        """

        dataset = self.dataset
        var = self.variavel

        dataset_2 = grade.dataset
        var_2 = grade.variavel

        if var == 'uwnd' and var_2 == 'vwnd':
            u = dataset[var]
            v = dataset_2[var_2]
        elif var_2 == 'uwnd' and var == 'vwnd':
            u = dataset_2[var_2]
            v = dataset[var]

        d = []

        dx, dy = mpcalc.lat_lon_grid_deltas(
            dataset.variables['lon'][:], dataset.variables['lat'][:])

        for i, data in enumerate(dataset.variables['time'][:]):
            div = mpcalc.divergence(
                u.isel(
                    time=i), v.isel(
                    time=i), dx, dy, dim_order='yx')

            d.append(
                xr.DataArray(
                    div.m,
                    dims=[
                        'lat',
                        'lon'],
                    coords={
                        'lat': dataset.variables['lat'][:],
                        'lon': dataset.variables['lon'][:],
                        'time': dataset.variables['time'][:][i]},
                    name='div'))

        divergence = xr.concat(d, dim='time').to_dataset()
        divergence.attrs = dataset.attrs

        return divergence


    def vorticidade(self, grade):
        """
        Retorna a vorticidade do vento.
        
        Args:
        grade: libs.grade.Grade
            Objeto libs.grade.Grade contendo um dataset com valores de vento na direção u ou v, complementar ao objeto Atmosfera.
        """
        dataset = self.dataset
        var = self.variavel

        dataset_2 = grade.dataset
        var_2 = grade.variavel

        if var == 'uwnd' and var_2 == 'vwnd':
            u = dataset[var]
            v = dataset_2[var_2]
        elif var_2 == 'uwnd' and var == 'vwnd':
            u = dataset_2[var_2]
            v = dataset[var]
        d = []

        dx, dy = mpcalc.lat_lon_grid_deltas(
            dataset.variables['lon'][:], dataset.variables['lat'][:])

        for i, data in enumerate(dataset.variables['time'][:]):
            vort = mpcalc.vorticity(
                u.isel(
                    time=i), v.isel(
                    time=i), dx, dy, dim_order='yx')

            d.append(
                xr.DataArray(
                    vort.m,
                    dims=[
                        'lat',
                        'lon'],
                    coords={
                        'lat': dataset.variables['lat'][:],
                        'lon': dataset.variables['lon'][:],
                        'time': dataset.variables['time'][:][i]},
                    name='vort'))

        vorticity = xr.concat(d, dim='time').to_dataset()
        vorticity.attrs = dataset.attrs

        return vorticity