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


archivo1 = fits.open('Medias-Desviacion279.fits')
series_ks_n_filtrado=np.array(archivo1[10].data)
errores_ks_n_filtrado=np.array(archivo1[11].data)
archivo = fits.open('TCampo_b279.fits')
HJD=archivo[1].data
archivo2= fits.open('PVariablesb2793.fits')
archivo3= fits.open('PVariablesb2792.fits')
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


def eliminar_nan(error, ks, HJD):
    indices_validos = ~np.isnan(ks)
    error_filtrado = [error[i] for i in range(len(ks)) if indices_validos[i]]
    ks_filtrado = [ks[i] for i in range(len(ks)) if indices_validos[i]]
    HJD_filtrado = [HJD[i] for i in range(len(ks)) if indices_validos[i]]
    return error_filtrado, ks_filtrado, np.round(HJD_filtrado,8)

ks_nan=[]
error_nan=[]
HJD_na=[]
y=0
while y<len(ks_aleatorios):
    error_filtrado, ks_filtrado, HJD_filtrado = eliminar_nan(errores_aleatorios[y], ks_aleatorios[y], HJD-2400000)
    ks_nan.append(ks_filtrado)
    error_nan.append(error_filtrado)
    HJD_na.append(HJD_filtrado)
    y+=1

x=0
while x<len(ks_nan):
    serie = ks_nan[x]
    erro = error_nan[x]
    hjd = HJD_na[x]

    # Definir el nombre del archivo de texto
    nombre_archivo = "D2b279"+str(x)+".txt"

    with open(nombre_archivo, "w") as archivo:
        for h, s, e in zip(hjd, serie, erro):
            archivo.write("{:<15} {:<10} {:<6}\n".format(h, s, e))
    x+=1
print("hecho")


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
hdul.writeto('T1b279_cluster.fits', overwrite=True)

