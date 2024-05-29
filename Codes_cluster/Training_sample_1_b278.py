#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
import os


# In[2]:


archivo1 = fits.open('Medias-Desviacion278.fits')
series_ks_n_filtrado=np.array(archivo1[9].data)
errores_ks_n_filtrado=np.array(archivo1[10].data)
archivo = fits.open('TCampob278.fits')
HJD=archivo[1].data
archivo2= fits.open('PVariablesb2783.fits')
archivo3= fits.open('PVariablesb2782.fits')
series_eliminar=np.array(archivo2[1].data)
series_eliminar1=np.array(archivo3[1].data)


# In[ ]:


def seleccionar_subarrays_aleatorios(series, errores, cantidad, series_eliminar):
    subarrays_elegidos = []
    errores_elegidos = []
    
    while len(subarrays_elegidos) < cantidad:
        indice_aleatorio = random.randint(0, len(series) - 1)
        if indice_aleatorio not in series_eliminar:
            if indice_aleatorio not in series_eliminar1:
                subarrays_elegidos.append(series[indice_aleatorio])
                errores_elegidos.append(errores[indice_aleatorio])
    
    return subarrays_elegidos, errores_elegidos

cantidad_subarrays = 100000
ks_aleatorios, errores_aleatorios = seleccionar_subarrays_aleatorios(series_ks_n_filtrado, errores_ks_n_filtrado, cantidad_subarrays, series_eliminar)
print(len(ks_aleatorios))


# In[ ]:


ks_aleatorios=np.array(ks_aleatorios)
errores_aleatorios=np.array(errores_aleatorios)
HJD=np.array(HJD)

# Crear un archivo FITS
hdul = fits.HDUList()  # Crear lista de HDUs

# Agregar el PrimaryHDU
primary_hdu = fits.PrimaryHDU()
hdul.append(primary_hdu)

# Crear extensiones con los arrays waves y flux
ks_hdu = fits.ImageHDU(ks_aleatorios, name='ks')
error_hdu = fits.ImageHDU(errores_aleatorios, name='EKS')
HJD_hdu = fits.ImageHDU(HJD, name='HJD')
# Agregar las extensiones a la lista de HDUs
hdul.append(ks_hdu)
hdul.append(error_hdu)
hdul.append(HJD_hdu)
# Guardar el archivo FITS
hdul.writeto('T1b278_cluster.fits', overwrite=True)

