#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle
import os
from statsmodels import robust
from statsmodels.robust.scale import huber
import math
from scipy.stats import linregress
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators import H2ORandomForestEstimator
from h2o.automl import H2OAutoML
import seaborn as sns
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
print('corri las librerias')

# In[ ]:


archivo = fits.open('TCampob278.fits') #Campo Completo 278
archivo5= fits.open('Features') #Campo Completo 279
archivo1 = fits.open('PVariablesb278.fits') #Primera muestra sobre el polinomio 103 en total
archivo2=fits.open('PVariablesb2783.fits') #Primera muestra debajo el polinomio 871 en total - Tercera muestra de entrenamiento 
archivo3=fits.open('PVariablesb2782.fits') #Segunda muestra de entrenamiento debajo del polinomio 857
archivo4 = fits.open('PVaribales_2b278.fits')
HJD=archivo[1].data #Campo Completo 278
HJD1=archivo5[5].data #Campo Completo 279
Error1=archivo1[4].data   #Primera muestra sobre el polinomio 103 en total
Ks1=archivo1[3].data      #Primera muestra sobre el polinomio 103 en total
Error2=archivo2[11].data   #Primera muestra debajo el polinomio 871 en total
Ks2=archivo2[10].data      #Primera muestra debajo el polinomio 871 en total
Error3=archivo3[4].data   #Segunda muestra debajo del polinomio 857
Ks3=archivo3[3].data      #Segunda muestra debajo del polinomio 857
Error4=archivo4[6].data   #Segunda muestra sobre el polinomio 111
Ks4=archivo4[5].data      #segunda muestra sobre el polinomio 111
Error5=archivo2[7].data   #Tercera muestra 854 en total
Ks5=archivo2[6].data      #Tercera muestra 854 en total
archivo3.info()
archivo2.info()
archivo1.info()
archivo4.info()

print('abri lo fits')
# In[ ]:


# Leer el primer archivo y cargar los datos en un DataFrame
d1_v = pd.read_csv('Cefeidas.txt', header=None, delim_whitespace=True)
d2_v = pd.read_csv('Binarias.txt', header=None, delim_whitespace=True)
d3_v = pd.read_csv('Lyrae.txt', header=None, delim_whitespace=True)
d4_v = pd.read_csv('LP.txt', header=None, delim_whitespace=True)
d5_v = pd.read_csv('Ninguna.txt', header=None, delim_whitespace=True)
d6_v = pd.read_csv('Ninguna278.txt', header=None, delim_whitespace=True)

print('abri los frame')
# In[ ]:


def data (inicio:str, Ks:np.array, nombre:str, variables, periodos):
    Datos1={}
    V1=[]
    for i, elemento in enumerate(variables):
        nombre_sin_extension = os.path.splitext(elemento)[0]  # Eliminar la extensión ".txt"
        if nombre_sin_extension.startswith(inicio):
            # Obtener el número final del nombre
            numero = int(nombre_sin_extension[len(inicio):])
            Datos1[nombre+str(numero)] = [periodos[i], Ks[numero]]
            V1.append(Ks[numero])
    return Datos1, V1

print('iniciare las listas')

#Cefeidas
data1, var1 = data('Datos1b278', Ks2, '1b278',d1_v[0].tolist(), d1_v[1].tolist())
data2, var2 = data('Datosb278', Ks1, '12b278', d1_v[0].tolist(), d1_v[1].tolist())
data3, var3 = data ('b278D', Ks4, '2b278', d1_v[0].tolist(), d1_v[1].tolist() )
data4, var4 = data ('Datos2b278', Ks3, '22b278', d1_v[0].tolist(), d1_v[1].tolist())
data5, var5 = data ('Datos3b278', Ks5, '3b278', d1_v[0].tolist(), d1_v[1].tolist())
data6,var6 = data('Datos1b279',Ks2, '1b279',d1_v[0].tolist(), d1_v[1].tolist())
data7, var7 = data('Datosb279', Ks1, '12b279', d1_v[0].tolist(), d1_v[1].tolist())
data8, var8 = data ('b279D', Ks4, '2b279', d1_v[0].tolist(), d1_v[1].tolist() )
data9, var9 = data ('Datos2b279', Ks3, '22b279', d1_v[0].tolist(), d1_v[1].tolist())
data10, var10= data ('Datos3b279', Ks5, '3b279', d1_v[0].tolist(), d1_v[1].tolist())
Cefeidas= {}
Cefeidas.update(data1)
Cefeidas.update(data2)
Cefeidas.update(data3)
Cefeidas.update(data4)
Cefeidas.update(data5)
cefeidas= var1+var2+var3+var4+var5
cefeidas1=var6+var7+var8+var9+var10

#Binarias
d1, v1 = data('Datos1b278', Ks2, '1b278',d2_v[0].tolist(), d2_v[1].tolist())
d2, v2 = data('Datosb278', Ks1, '12b278', d2_v[0].tolist(), d2_v[1].tolist())
d3, v3 = data ('b278D', Ks4, '2b278', d2_v[0].tolist(), d2_v[1].tolist() )
d4, v4 = data ('Datos2b278', Ks3, '22b278', d2_v[0].tolist(), d2_v[1].tolist())
d5, v5 = data ('Datos3b278', Ks5, '3b278', d2_v[0].tolist(), d2_v[1].tolist())
d6,v6 = data('Datos1b279',Ks2, '1b279',d2_v[0].tolist(), d2_v[1].tolist())
d7, v7 = data('Datosb279', Ks1, '12b279', d2_v[0].tolist(), d2_v[1].tolist())
d8, v8 = data ('b279D', Ks4, '2b279', d2_v[0].tolist(), d2_v[1].tolist() )
d9, v9 = data ('Datos2b279', Ks3, '22b279', d2_v[0].tolist(), d2_v[1].tolist())
d10, v10= data ('Datos3b279', Ks5, '3b279', d2_v[0].tolist(), d2_v[1].tolist())
Binarias= {}
Binarias.update(d1)
Binarias.update(d2)
Binarias.update(d3)
Binarias.update(d4)
Binarias.update(d5)
binarias= v1+v2+v3+v4+v5
binarias1=v6+v7+v8+v9+v10


#Lyrae
da1, va1 = data('Datos1b278', Ks2, '1b278',d3_v[0].tolist(), d3_v[1].tolist())
da2, va2 = data('Datosb278', Ks1, '12b278', d3_v[0].tolist(), d3_v[1].tolist())
da3, va3 = data ('b278D', Ks4, '2b278', d3_v[0].tolist(), d3_v[1].tolist() )
da4, va4 = data ('Datos2b278', Ks3, '22b278', d3_v[0].tolist(), d3_v[1].tolist())
da5, va5 = data ('Datos3b278', Ks5, '3b278', d3_v[0].tolist(), d3_v[1].tolist())
da6, va6 = data('Datos1b279', Ks2, '1b279',d3_v[0].tolist(), d3_v[1].tolist())
da7, va7 = data('Datosb279', Ks1, '12b279', d3_v[0].tolist(), d3_v[1].tolist())
da8, va8 = data ('b279D', Ks4, '2b279', d3_v[0].tolist(), d3_v[1].tolist() )
da9, va9 = data ('Datos2b279', Ks3, '22b279', d3_v[0].tolist(), d3_v[1].tolist())
da10,va10= data ('Datos3b279', Ks5, '3b279', d3_v[0].tolist(), d3_v[1].tolist())

Lyrae= {}
Lyrae.update(da1)
Lyrae.update(da2)
Lyrae.update(da3)
Lyrae.update(da4)
Lyrae.update(da5)
lyrae= va1+va2+va3+va4+va5
lyrae1=va6+va7+va8+va9+va10

#LPV
ata1, ar1 = data('Datos1b278', Ks2, '1b278',d4_v[0].tolist(), d4_v[1].tolist())
ata2, ar2 = data('Datosb278', Ks1, '12b278', d4_v[0].tolist(), d4_v[1].tolist())
ata3, ar3 = data ('b278D', Ks4, '2b278', d4_v[0].tolist(), d4_v[1].tolist() )
ata4, ar4 = data ('Datos2b278', Ks3, '22b278', d4_v[0].tolist(), d4_v[1].tolist())
ata5, ar5 = data ('Datos3b278', Ks5, '3b278', d4_v[0].tolist(), d4_v[1].tolist())
ata6, ar6 = data('Datos1b279', Ks2, '1b279',d4_v[0].tolist(), d4_v[1].tolist())
ata7, ar7 = data('Datosb279', Ks1, '12b279', d4_v[0].tolist(), d4_v[1].tolist())
#ata8, ar8 = data ('b279D', Ks4, '2b279',  d4_v[0].tolist(), d4_v[1].tolist())
ata9, ar9 = data ('Datos2b279', Ks3, '22b279', d4_v[0].tolist(), d4_v[1].tolist())
ata10,ar10= data ('Datos3b279', Ks5, '3b279', d4_v[0].tolist(), d4_v[1].tolist())
LPV= {}
LPV.update(ata1)
LPV.update(ata2)
LPV.update(ata3)
LPV.update(ata4)
LPV.update(ata5)
lpv=ar1+ar2+ar3+ar4+ar5
lpv1= ar6+ar7+ar9+ar10
#Ninguna
nd, No = data ('Datos3b278', Ks5, '3b278', d6_v[0].tolist(), d6_v[1].tolist())
nd1, No1 = data ('Datos3b279', Ks5, '3b279', d5_v[0].tolist(), d5_v[1].tolist())
print(len(cefeidas))
print(len(binarias))
print(len(lyrae))
print(len(lpv))
print(len(No))


# In[ ]:


def estandarizar_lista(lista):
    # Convertir la lista a un array de NumPy
    lista = np.array(lista)
    
    # Encontrar las posiciones de NaNs en la lista
    nan_indices = np.isnan(lista)
    
    # Calcular la media y la desviación estándar de la lista sin NaNs
    media = np.nanmean(lista)
    desviacion_estandar = np.nanstd(lista)
    
    # Estandarizar la lista
    lista_estandarizada = (lista - media) / desviacion_estandar
    
    # Reemplazar los NaNs en la lista estandarizada con ceros
    lista_estandarizada[np.isnan(lista_estandarizada)] = 0
    
    return lista_estandarizada


# # 1. MAD

# Función para calcular la Mediana de la Desviación Absoluta (MAD) eliminando valores np.nan
def mad(arr1):
    mads_arr1 = []
    for sublist in arr1:
        arr_cleaned = [x for x in sublist if not np.isnan(x)]
        if arr_cleaned:  # Verifica que la lista no esté vacía después de eliminar np.nan
            median = np.median(arr_cleaned)
            mad = np.median(np.abs(arr_cleaned - median))
            mads_arr1.append(mad)
    
    return mads_arr1
# Calcular MAD para los datos
mad_cefeidas=mad(cefeidas)
mad_binarias=mad(binarias)
mad_lyrae=mad(lyrae)
mad_lpv=mad(lpv)
mad_no=mad(No)
#Estandarizar
mad_cefeidas=np.array(estandarizar_lista(mad_cefeidas))
mad_binarias=np.array(estandarizar_lista(mad_binarias))
mad_lyrae=np.array(estandarizar_lista(mad_lyrae))
mad_lpv=np.array(estandarizar_lista(mad_lpv))
mad_no=np.array(estandarizar_lista(mad_no))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(mad_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(mad_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(mad_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(mad_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(mad_no, bins=10,  alpha=0.5, density=True, label='None')
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('MAD')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('CMAD.png')
# Mostrar la gráfica
plt.show()
plt.close()
print("termine de calcular Mad")




# 2. Mediana

def calcular_mediana_por_objeto(s_variables):
    # Lista para almacenar los resultados
    resultados_variables = []

    # Iterar sobre las sub listas en s_variables
    for sublist in s_variables:
        # Filtrar np.nan de la sub lista
        sublist_filtrada = [x for x in sublist if not np.isnan(x)]
        # Calcular la mediana
        mediana = np.median(sublist_filtrada)
        # Agregar la mediana a los resultados
        resultados_variables.append(mediana)
    return resultados_variables
# Calcular Mediana para los datos
mean_cefeidas=calcular_mediana_por_objeto(cefeidas)
mean_binarias=calcular_mediana_por_objeto(binarias)
mean_lyrae=calcular_mediana_por_objeto(lyrae)
mean_lpv=calcular_mediana_por_objeto(lpv)
mean_no=calcular_mediana_por_objeto(No)
#Estandarizar
mean_cefeidas=np.array(estandarizar_lista(mean_cefeidas))
mean_binarias=np.array(estandarizar_lista(mean_binarias))
mean_lyrae=np.array(estandarizar_lista(mean_lyrae))
mean_lpv=np.array(estandarizar_lista(mean_lpv))
mean_no=np.array(estandarizar_lista(mean_no))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(mean_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(mean_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(mean_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(mean_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(mean_no, bins=10,  alpha=0.5, density=True, label='None')
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('Mean')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('CMean.png')
# Mostrar la gráfica
plt.show()
plt.close()
print("Calcule Media")


# 3. VART

def calcular_VART(sublista, tiempo):
    # Eliminar valores np.nan de la sublista y del tiempo
    sublista_limpia = [valor for valor in sublista if not np.isnan(valor)]
    tiempo_limpio = [t for t, valor in zip(tiempo, sublista) if not np.isnan(valor)]
    # Calcular la mediana de la sublista
    mediana = np.median(sublista_limpia)
    # Calcular el VART según la fórmula
    VART = np.sum(np.abs(np.array(sublista_limpia) - mediana) / np.array(tiempo_limpio))
    return VART
# Calcular VART para cada sublista en s_variables y s_novariables
VART_cefeidas= [calcular_VART(sublista, HJD) for sublista in cefeidas]
VART_binarias = [calcular_VART(sublista, HJD) for sublista in binarias]
VART_lyrae = [calcular_VART(sublista, HJD) for sublista in lyrae]
VART_lpv = [calcular_VART(sublista, HJD) for sublista in lpv]
VART_No = [calcular_VART(sublista, HJD) for sublista in No]
#Estandarizar
VART_cefeidas=np.array(estandarizar_lista(VART_cefeidas))
VART_binarias=np.array(estandarizar_lista(VART_binarias))
VART_lyrae=np.array(estandarizar_lista(VART_lyrae))
VART_lpv=np.array(estandarizar_lista(VART_lpv))
VART_No=np.array(estandarizar_lista(VART_No))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(VART_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(VART_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(VART_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(VART_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(VART_No, bins=10,  alpha=0.5, density=True, label='None')
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('VART')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('CVART.png')
# Mostrar la gráfica
plt.show()
plt.close()
print("Calcule VART")


# 4. Factor AV

def calcular_av(lista):
    # Eliminar np.nan de la lista
    lista_sin_nan = [x for x in lista if not np.isnan(x)]
    # Calcular la media de la lista
    media = np.mean(lista_sin_nan)
    # Calcular el numerador de la fórmula
    numerador = np.sum(np.diff(lista_sin_nan)**2)
    # Calcular el denominador de la fórmula
    denominador = 2 * (len(lista_sin_nan) - 1) * np.sum((lista_sin_nan - media)**2)
    # Calcular el valor Abbe
    if denominador != 0:
        av = numerador / denominador
    else:
        av = np.nan
    return av
def calcular_av_para_listas_de_listas(lista_de_listas):
    resultados = []
    for sublista in lista_de_listas:
        av = calcular_av(sublista)
        resultados.append(av)
    return resultados
av_cefeidas= calcular_av_para_listas_de_listas(cefeidas)
av_binarias= calcular_av_para_listas_de_listas(binarias)
av_lyrae= calcular_av_para_listas_de_listas(lyrae)
av_lpv =calcular_av_para_listas_de_listas(lpv)
av_No =calcular_av_para_listas_de_listas(No)
#Estandarizar
av_cefeidas=np.array(estandarizar_lista(VART_cefeidas))
av_binarias=np.array(estandarizar_lista(VART_binarias))
av_lyrae=np.array(estandarizar_lista(VART_lyrae))
av_lpv=np.array(estandarizar_lista(VART_lpv))
av_No=np.array(estandarizar_lista(VART_No))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(av_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(av_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(av_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(av_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(av_No, bins=10,  alpha=0.5, density=True, label='None')
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('AV Factor')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('CAV.png')
# Mostrar la gráfica
plt.show()
plt.close()
print("Calcule Vart")



# 5. robAbbe

def huber_estimator(data, c=1.345):
    median = np.median(data)
    diff = np.abs(data - median)
    outlier_mask = diff > c * np.median(diff)
    data_clean = data[~outlier_mask]
    return np.median(data_clean), np.median(np.abs(data_clean - np.median(data_clean))) * 1.4826
def calcular_robAbbe(sublist, c=1.345):
    # Encontrar índices de np.nan
    nan_indices = np.where(np.isnan(sublist))[0]
    # Eliminar los valores np.nan de la sublist
    sublist_clean = np.array([x for x in sublist if not np.isnan(x)])
    # Calcular la M-estimación de Huber para la sublist
    _, mad = huber_estimator(sublist_clean, c=c)
    # Restaurar los np.nan a sus posiciones originales
    robAbbe = np.empty(len(sublist))
    robAbbe.fill(np.nan)
    robAbbe[np.where(np.isnan(sublist))] = np.nan
    robAbbe[np.where(~np.isnan(sublist))] = mad
    return robAbbe
def calcular_robAbbe_separado(s_variables, c=1.345):
    resultados_s_variables = []
    resultados_s_novariables = []
    for sublist_vars in s_variables:
        robAbbe_vars = calcular_robAbbe(sublist_vars, c=c)
        resultados_s_variables.append(np.nanmean(robAbbe_vars))  # Agregar la media de los resultados
    return resultados_s_variables

rob_cefeidas= calcular_robAbbe_separado(cefeidas)
rob_binarias= calcular_robAbbe_separado(binarias)
rob_lyrae= calcular_robAbbe_separado(lyrae)
rob_lpv =calcular_robAbbe_separado(lpv)
rob_No =calcular_robAbbe_separado(No)
#Estandarizar
rob_cefeidas=np.array(estandarizar_lista(rob_cefeidas))
rob_binarias=np.array(estandarizar_lista(rob_binarias))
rob_lyrae=np.array(estandarizar_lista(rob_lyrae))
rob_lpv=np.array(estandarizar_lista(rob_lpv))
rob_No=np.array(estandarizar_lista(rob_No))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(rob_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(rob_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(rob_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(rob_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(rob_No, bins=10,  alpha=0.5, density=True, label='None')
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('robAbbe')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('CrobAbbe.png')
# Mostrar la gráfica
plt.show()
plt.close()
print("Calcule rob")


# 6. DIFDER

def calcular_DIFDER(X, Y):
    N = len(X)
    DIFDER_resultados = []    
    for Y_i in Y:
        suma_difder = 0
        count = 0
        for i in range(N-2):
            if not (np.isnan(X[i]) or np.isnan(X[i+1]) or np.isnan(X[i+2])):
                numerador1 = Y_i[i] - Y_i[i+1]
                denominador1 = X[i] - X[i+1]
                numerador2 = Y_i[i+1] - Y_i[i+2]
                denominador2 = X[i+1] - X[i+2]
                if denominador1 != 0 and denominador2 != 0:
                    difder = ((numerador1 / denominador1) - (numerador2 / denominador2)) - (numerador1 / denominador1)
                    suma_difder += difder
                    count += 1
        if count != 0:
            DIFDER_resultados.append(suma_difder / count)
        else:
            DIFDER_resultados.append(np.nan)    
    return DIFDER_resultados


DIFDER_cefeidas= calcular_DIFDER(HJD, cefeidas)
DIFDER_binarias=calcular_DIFDER(HJD, binarias)
DIFDER_lyrae= calcular_DIFDER(HJD, lyrae)
DIFDER_lpv =calcular_DIFDER(HJD, lpv)
DIFDER_No =calcular_DIFDER(HJD, No)
#Estandarizar
DIFDER_cefeidas=np.array(estandarizar_lista(DIFDER_cefeidas))
DIFDER_binarias=np.array(estandarizar_lista(DIFDER_binarias))
DIFDER_lyrae=np.array(estandarizar_lista(DIFDER_lyrae))
DIFDER_lpv=np.array(estandarizar_lista(DIFDER_lpv))
DIFDER_No=np.array(estandarizar_lista(DIFDER_No))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(DIFDER_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(DIFDER_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(DIFDER_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(DIFDER_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(DIFDER_No, bins=10,  alpha=0.5, density=True, label='None')
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('DIFDER')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('CDIFDE.png')
# Mostrar la gráfica
plt.show()
plt.close()
print("calcule DIFER")



# 7. PROY 2
def calcular_proyecciones(HJD, s_variables):
    # Función para eliminar los valores nan de una lista
    def remove_nan(lst):
        return [x for x in lst if not np.isnan(x)]
    
    # Eliminar los valores nan de las listas en s_variables y guardar en HDJ_variables
    variables = [remove_nan(lst) for lst in s_variables]
    
    # Función para eliminar np.nan de HJD por cada sublista de s_variables
    def remove_nan_in_HJD(HJD, s_variables):
        result = []
        for sublist in s_variables:
            valid_indices = [i for i, val in enumerate(sublist) if not np.isnan(val)]
            result.append([HJD[i] for i in valid_indices])
        return result
    
    # Eliminar np.nan en HJD por cada sublista de s_variables
    HJD_variable = remove_nan_in_HJD(HJD, s_variables)
    
    # Función para calcular la proyección según la fórmula dada
    def calcular_proyeccion(X, Y):
        N = len(X)
        proyecciones = []
        for i in range(1, N-1):
            proyeccion = ((Y[i+1] - Y[i-1]) / (X[i+1] - X[i-1])) * (X[i] - X[i-1]) + Y[i-1] - Y[i]
            proyecciones.append(proyeccion)
        return proyecciones
    
    # Calcular proyección para cada sublista y guardar los resultados en una lista
    proy_varibles = []
    for i in range(len(HJD_variable)):
        X_actual = HJD_variable[i]
        Y_actual = variables[i]
        proyecciones_actual = calcular_proyeccion(X_actual, Y_actual)
        proyeccion_promedio = sum(proyecciones_actual) / len(proyecciones_actual)
        proy_varibles.append(proyeccion_promedio)
    
    return proy_varibles

# Uso de la función con s_variables y HJD, ignorando s_novariables
PROY_cefeidas= calcular_proyecciones(HJD, cefeidas)
PROY_binarias=calcular_proyecciones(HJD, binarias)
PROY_lyrae= calcular_proyecciones(HJD, lyrae)
PROY_lpv =calcular_proyecciones(HJD, lpv)
PROY_No =calcular_proyecciones(HJD, No)
#Estandarizar
PROY_cefeidas=np.array(estandarizar_lista(PROY_cefeidas))
PROY_binarias=np.array(estandarizar_lista(PROY_binarias))
PROY_lyrae=np.array(estandarizar_lista(PROY_lyrae))
PROY_lpv=np.array(estandarizar_lista(PROY_lpv))
PROY_No=np.array(estandarizar_lista(PROY_No))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(PROY_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(PROY_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(PROY_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(PROY_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(PROY_No, bins=10,  alpha=0.5, density=True, label='None') 
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('PROY 2')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('CPROY2.png')
# Mostrar la gráfica
plt.show()
plt.close()
print("termine proy2")




# 8. INTEGRAL 
def calcular_integrales(HJD, s_variables):
    # Función para eliminar los valores nan de una lista
    def remove_nan(lst):
        return [x for x in lst if not np.isnan(x)]
    # Eliminar los valores nan de las listas en s_variables y guardar en HDJ_variables
    variables = [remove_nan(lst) for lst in s_variables]
        # Función para eliminar np.nan de HJD por cada sublista de s_variables
    def remove_nan_in_HJD(HJD, s_variables):
        result = []
        for sublist in s_variables:
            valid_indices = [i for i, val in enumerate(sublist) if not np.isnan(val)]
            result.append([HJD[i] for i in valid_indices])
        return result
        # Eliminar np.nan en HJD por cada sublista de s_variables
    HJD_variable = remove_nan_in_HJD(HJD, s_variables)
        # Función para calcular la integral según la fórmula dada
    def calcular_integral(X, Y):
        N = len(X)
        delta_X = [X[i + 1] - X[i] for i in range(N - 1)]
        mean_Y = sum(Y) / N
        integral = sum([(delta_X[i] * (Y[i] - mean_Y)) for i in range(N - 1)]) / (X[N - 1] - X[0])
        return integral
        # Calcular integral para cada sublista y guardar los resultados en una lista
    integrals_variables = []
    for i in range(len(HJD_variable)):
        integral = calcular_integral(HJD_variable[i], variables[i])
        integrals_variables.append(abs(integral))
    return integrals_variables

# Uso de la función con s_variables y HJD, ignorando s_novariables
integrals_cefeidas= calcular_integrales(HJD, cefeidas)
integrals_binarias=calcular_integrales(HJD, binarias)
integrals_lyrae= calcular_integrales(HJD, lyrae)
integrals_lpv =calcular_integrales(HJD, lpv)
integrals_No =calcular_integrales(HJD, No)
#Estandarizar
integrals_cefeidas=np.array(estandarizar_lista(integrals_cefeidas))
integrals_binarias=np.array(estandarizar_lista(integrals_binarias))
integrals_lyrae=np.array(estandarizar_lista(integrals_lyrae))
integrals_lpv=np.array(estandarizar_lista(integrals_lpv))
integrals_No=np.array(estandarizar_lista(integrals_No))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(integrals_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(integrals_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(integrals_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(integrals_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(integrals_No, bins=10,  alpha=0.5, density=True, label='None') 
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('INTEGRAL')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('CIntegral.png')
# Mostrar la gráfica
plt.show()
plt.close()
print("calcule Integral")



# 10. RULD

def calcular_RULDs(HJD, s_variables):
    # Función para eliminar los valores nan de una lista
    def remove_nan(lst):
        return [x for x in lst if not np.isnan(x)]
    
    # Eliminar los valores nan de las listas en s_variables y guardar en HDJ_variables
    variables = [remove_nan(lst) for lst in s_variables]
    
    # Función para eliminar np.nan de HJD por cada sublista de s_variables
    def remove_nan_in_HJD(HJD, s_variables):
        result = []
        for sublist in s_variables:
            valid_indices = [i for i, val in enumerate(sublist) if not np.isnan(val)]
            result.append([HJD[i] for i in valid_indices])
        return result
    
    # Eliminar np.nan en HJD por cada sublista de s_variables
    HJD_variable = remove_nan_in_HJD(HJD, s_variables)
    
    # Función para calcular el RULD según la fórmula dada
    def calcular_RULD(X, Y):
        N = len(X)
        ruld_sum = sum([((Y[i] - Y[i + 1]) / abs(Y[i] - Y[i + 1])) * (X[i + 1] - X[i]) for i in range(N - 1)])
        ruld = (1 / (N - 1)) * ruld_sum
        return ruld
    
    # Calcular RULD para cada sublista y guardar los resultados en una lista
    rulds_variables = []
    for i in range(len(HJD_variable)):
        ruld = calcular_RULD(HJD_variable[i], variables[i])
        rulds_variables.append(ruld)
    return rulds_variables
# Uso de la función con s_variables y HJD, ignorando s_novariables
RULD_cefeidas= calcular_RULDs(HJD, cefeidas)
RULD_binarias=calcular_RULDs(HJD, binarias)
RULD_lyrae= calcular_RULDs(HJD, lyrae)
RULD_lpv =calcular_RULDs(HJD, lpv)
RULD_No =calcular_RULDs(HJD, No)
#Estandarizar
RULD_cefeidas=np.array(estandarizar_lista(RULD_cefeidas))
RULD_binarias=np.array(estandarizar_lista(RULD_binarias))
RULD_lyrae=np.array(estandarizar_lista(RULD_lyrae))
RULD_lpv=np.array(estandarizar_lista(RULD_lpv))
RULD_No=np.array(estandarizar_lista(RULD_No))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(RULD_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(RULD_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(RULD_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(RULD_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(RULD_No, bins=10,  alpha=0.5, density=True, label='None') 
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('RULD')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('CRULD.png')
# Mostrar la gráfica
plt.show()
plt.close()
print("RULD")


# 11. Asimetria de octil (OS)
# Función para calcular la asimetría de octil (OS)
def calcular_os(lista):
    q1 = np.percentile(lista, 25)
    q3 = np.percentile(lista, 75)
    os = 2 * (q3 - q1) / (q3 + q1)
    return os
def calcular_OS(s_variables):
    # Función para eliminar los valores nan de una lista
    def remove_nan(lst):
        return [x for x in lst if not np.isnan(x)]
    # Eliminar los valores nan de las listas en s_variables y guardar en variables
    variables = [remove_nan(lst) for lst in s_variables]
    # Lista para almacenar los resultados de OS
    os_resultados = []
    # Calcular OS para cada sublista en "variables" y guardar los resultados
    for sublist in variables:
        os_sublist = calcular_os(sublist)
        os_resultados.append(os_sublist)
    return os_resultados

# Uso de la función con s_variables y HJD, ignorando s_novariables
os_cefeidas= calcular_OS(cefeidas)
os_binarias=calcular_OS(binarias)
os_lyrae= calcular_OS(lyrae)
os_lpv=calcular_OS(lpv)
os_No=calcular_OS(No)
#Estandarizar
os_cefeidas=np.array(estandarizar_lista(os_cefeidas))
os_binarias=np.array(estandarizar_lista(os_binarias))
os_lyrae=np.array(estandarizar_lista(os_lyrae))
os_lpv=np.array(estandarizar_lista(os_lpv))
os_No=np.array(estandarizar_lista(os_No))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(os_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(os_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(os_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(os_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(os_No, bins=10,  alpha=0.5, density=True, label='None') 
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('OS')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('OS.png')
# Mostrar la gráfica
plt.show()
plt.close()
print("calcule os")


# 12. REUCLID

def calcular_distancias_euclidianas(s_variables, HJD):
    # Función para eliminar los valores nan de una lista
    def remove_nan(lst):
        return [x for x in lst if not np.isnan(x)]
    
    # Eliminar los valores nan de las listas en s_variables y guardar en HDJ_variables
    variables = [remove_nan(lst) for lst in s_variables]
    
    # Función para eliminar np.nan de HJD por cada sublista de s_variables
    def remove_nan_in_HJD(HJD, s_variables):
        result = []
        for sublist in s_variables:
            valid_indices = [i for i, val in enumerate(sublist) if not np.isnan(val)]
            result.append([HJD[i] for i in valid_indices])
        return result
    
    # Eliminar np.nan en HJD por cada sublista de s_variables
    HJD_variable = remove_nan_in_HJD(HJD, s_variables)
    
    # Función para calcular la distancia euclidiana según la fórmula dada
    def calcular_distancia_euclidiana(X, Y):
        N = len(X)
        distancias = [math.sqrt((X[i + 1] - X[i])**2 + (Y[i + 1] - Y[i])**2) for i in range(N - 1)]
        distancia_media = sum(distancias) / (N - 1)
        return distancia_media
    
    # Calcular distancia euclidiana para cada sublista y guardar los resultados en una lista
    rEucliDs_variables = []
    for i in range(len(HJD_variable)):
        rEucliD = calcular_distancia_euclidiana(HJD_variable[i], variables[i])
        rEucliDs_variables.append(rEucliD)
    
    return rEucliDs_variables

# Uso de la función con s_variables y HJD, ignorando s_novariables
rEucliDs_cefeidas= calcular_distancias_euclidianas(cefeidas, HJD)
rEucliDs_binarias=calcular_distancias_euclidianas(binarias, HJD)
rEucliDs_lyrae= calcular_distancias_euclidianas(lyrae, HJD)
rEucliDs_lpv=calcular_distancias_euclidianas(lpv, HJD)
rEucliDs_No=calcular_distancias_euclidianas(No, HJD)
#Estandarizar
rEucliDs_cefeidas=np.array(estandarizar_lista(rEucliDs_cefeidas))
rEucliDs_binarias=np.array(estandarizar_lista(rEucliDs_binarias))
rEucliDs_lyrae=np.array(estandarizar_lista(rEucliDs_lyrae))
rEucliDs_lpv=np.array(estandarizar_lista(rEucliDs_lpv))
rEucliDs_No=np.array(estandarizar_lista(rEucliDs_No))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(rEucliDs_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(rEucliDs_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(rEucliDs_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(rEucliDs_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(rEucliDs_No, bins=10,  alpha=0.5, density=True, label='None') 
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('REUCLIDS')
ax.set_xlabel('Values')
ax.set_ylabel('Density')

# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('CREUCLIDS.png')
# Mostrar la gráfica
plt.show()
plt.close()
print("calcule reuclids")



# 13. Low

def calcular_low(s_variables, HDJ):
    # Lista para almacenar los resultados de la característica "low"
    resultados_low = []
    
    for sublist in s_variables:
        # Calcular el rango intercuartílico
        q25 = np.percentile(sublist, 25)
        q12_5 = np.percentile(sublist, 12.5)
        
        # Calcular la característica "low"
        low = ((q25 - q12_5) - (q12_5 - np.percentile(sublist, 0))) / (q25 - np.percentile(sublist, 0))
        
        # Agregar el resultado a la lista de resultados
        resultados_low.append(low)
    
    return resultados_low

# Uso de la función con s_variables y HJD, ignorando s_novariables
low_cefeidas= calcular_low(cefeidas, HJD)
low_binarias=calcular_low(binarias, HJD)
low_lyrae= calcular_low(lyrae, HJD)
low_lpv=calcular_low(lpv, HJD)
low_No=calcular_low(No, HJD)
#Estandarizar
low_cefeidas=np.array(estandarizar_lista(low_cefeidas))
low_binarias=np.array(estandarizar_lista(low_binarias))
low_lyrae=np.array(estandarizar_lista(low_lyrae))
low_lpv=np.array(estandarizar_lista(low_lpv))
low_No=np.array(estandarizar_lista(low_No))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(low_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(low_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(low_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(low_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(low_No, bins=10,  alpha=0.5, density=True, label='None') 
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('low')
ax.set_xlabel('Values')
ax.set_ylabel('Density')

# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('CLow.png')
# Mostrar la gráfica
plt.show()
plt.close()
print("calcule low")

#14. Row

def calcular_row(s_variables, HDJ):
    # Lista para almacenar los resultados de la característica "row"
    resultados_row = []   
    for sublist in s_variables:
        # Calcular el rango intercuartílico
        q75 = np.percentile(sublist, 75)
        q62_5 = np.percentile(sublist, 62.5)
        
        # Calcular la característica "row"
        row = ((q75 - q62_5) - (q62_5 - np.percentile(sublist, 50))) / (q75 - np.percentile(sublist, 50))
        
        # Agregar el resultado a la lista de resultados
        resultados_row.append(row)
    
    return resultados_row

# Uso de la función con s_variables y HJD, ignorando s_novariables
row_cefeidas= calcular_row(cefeidas, HJD)
row_binarias=calcular_row(binarias, HJD)
row_lyrae= calcular_row(lyrae, HJD)
row_lpv=calcular_row(lpv, HJD)
row_No=calcular_row(No, HJD)
#Estandarizar
row_cefeidas=np.array(estandarizar_lista(row_cefeidas))
row_binarias=np.array(estandarizar_lista(row_binarias))
row_lyrae=np.array(estandarizar_lista(row_lyrae))
row_lpv=np.array(estandarizar_lista(row_lpv))
row_No=np.array(estandarizar_lista(row_No))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(row_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(row_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(row_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(row_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(row_No, bins=10,  alpha=0.5, density=True, label='None') 
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('row')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('CRow.png')
# Mostrar la gráfica
plt.show()
plt.close()
print("calcule row")



# 15. DeltaM

def calcular_DeltaM(s_variables, HDJ):
    # Lista para almacenar los resultados de la característica "DeltaM"
    resultados_DeltaM = []
    
    for sublist in s_variables:
        # Calcular la característica "DeltaM"
        frac = []
        for i in range(len(HDJ) - 1):
            intervalo = [dato for dato in sublist if HDJ[i] <= dato <= HDJ[i + 1]]
            if intervalo:
                upperF = np.percentile(intervalo, 98)
                lowerF = np.percentile(intervalo, 2)
                frac.append(upperF - lowerF)
        
        # Calcular la mediana de las diferencias
        if frac:
            DeltaM = np.nanmedian(frac)
        else:
            DeltaM = np.nan  # Si no hay datos en el intervalo, asignar NaN
        
        # Agregar el resultado a la lista de resultados
        resultados_DeltaM.append(DeltaM)
    
    return resultados_DeltaM

# Uso de la función con s_variables y HJD, ignorando s_novariables
DeltaM_cefeidas= calcular_DeltaM(cefeidas, HJD)
DeltaM_binarias=calcular_DeltaM(binarias, HJD)
DeltaM_lyrae= calcular_DeltaM(lyrae, HJD)
DeltaM_lpv=calcular_DeltaM(lpv, HJD)
DeltaM_No=calcular_DeltaM(No, HJD)
#Estandarizar
DeltaM_cefeidas=np.array(estandarizar_lista(DeltaM_cefeidas))
DeltaM_binarias=np.array(estandarizar_lista(DeltaM_binarias))
DeltaM_lyrae=np.array(estandarizar_lista(DeltaM_lyrae))
DeltaM_lpv=np.array(estandarizar_lista(DeltaM_lpv))
DeltaM_No=np.array(estandarizar_lista(DeltaM_No))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(DeltaM_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(DeltaM_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(DeltaM_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(DeltaM_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(DeltaM_No, bins=10,  alpha=0.5, density=True, label='None') 
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('DeltaM')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('CDeltaM.png')
# Mostrar la gráfica
plt.show()
plt.close()
print("calcule delta")



# 16. Slope
def calcular_slope(s_variables, HJD):
    # Lista para almacenar los resultados de la característica "slope"
    resultados_slope = []
    
    for sublist in s_variables:
        # Calcular la regresión lineal
        slope, intercept, r_value, p_value, std_err = linregress(HJD, sublist)
        
        # Agregar el resultado a la lista de resultados
        resultados_slope.append(slope)
    
    return resultados_slope

# Uso de la función con s_variables y HJD, ignorando s_novariables
slope_cefeidas= calcular_slope(cefeidas, HJD)
slope_binarias=calcular_slope(binarias, HJD)
slope_lyrae= calcular_slope(lyrae, HJD)
slope_lpv=calcular_slope(lpv, HJD)
slope_No=calcular_slope(No, HJD)
#Estandarizar
slope_cefeidas=np.array(estandarizar_lista(slope_cefeidas))
slope_binarias=np.array(estandarizar_lista(slope_binarias))
slope_lyrae=np.array(estandarizar_lista(slope_lyrae))
slope_lpv=np.array(estandarizar_lista(slope_lpv))
slope_No=np.array(estandarizar_lista(slope_No))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(slope_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(slope_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(slope_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(slope_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(slope_No, bins=10,  alpha=0.5, density=True, label='None') 

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('Slope')
ax.set_xlabel('Values')# Uso de la función con s_variables y HJD, ignorando s_novariables
slope_cefeidas= calcular_slope(cefeidas, HJD)
slope_binarias=calcular_slope(binarias, HJD)
slope_lyrae= calcular_slope(lyrae, HJD)
slope_lpv=calcular_slope(lpv, HJD)
slope_No=calcular_slope(No, HJD)
#Estandarizar
slope_cefeidas=np.array(estandarizar_lista(slope_cefeidas))
slope_binarias=np.array(estandarizar_lista(slope_binarias))
slope_lyrae=np.array(estandarizar_lista(slope_lyrae))
slope_lpv=np.array(estandarizar_lista(slope_lpv))
slope_No=np.array(estandarizar_lista(slope_No))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(slope_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(slope_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(slope_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(slope_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(slope_No, bins=10,  alpha=0.5, density=True, label='None') 

ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('CSlope.png')
# Mostrar la gráfica
plt.show()
plt.close()

print("calcule slope")



# 17. Slope_min

def calcular_slope_min(s_variables, HJD):
    # Lista para almacenar los resultados de la característica "slope_min"
    resultados_slope_min = []
    
    for sublist in s_variables:
        # Inicializar la lista para almacenar las pendientes de las regresiones lineales
        slopes = []
        
        # Calcular la regresión lineal para cada subintervalo
        for i in range(len(HJD) - 1):
            # Seleccionar los datos dentro del intervalo de tiempo
            x = []
            y = []
            for j in range(len(HJD)):
                if HJD[i] <= HJD[j] <= HJD[i + 1]:
                    x.append(HJD[j])
                    y.append(sublist[j])
            
            # Calcular la regresión lineal para el subintervalo
            if len(x) > 1:
                slope, _, _, _, _ = linregress(x, y)
                slopes.append(slope)
        
        # Calcular el mínimo de las pendientes de las regresiones lineales
        if slopes:
            slope_min = min(slopes)
        else:
            slope_min = None
        
        # Agregar el resultado a la lista de resultados
        resultados_slope_min.append(slope_min)
    
    return resultados_slope_min

# Uso de la función con s_variables y HJD, ignorando s_novariables
slopemin_cefeidas= calcular_slope_min(cefeidas, HJD)
slopemin_binarias=calcular_slope_min(binarias, HJD)
slopemin_lyrae= calcular_slope_min(lyrae, HJD)
slopemin_lpv=calcular_slope_min(lpv, HJD)
slopemin_No=calcular_slope_min(No, HJD)
#Estandarizar
slopemin_cefeidas=np.array(estandarizar_lista(slopemin_cefeidas))
slopemin_binarias=np.array(estandarizar_lista(slopemin_binarias))
slopemin_lyrae=np.array(estandarizar_lista(slopemin_lyrae))
slopemin_lpv=np.array(estandarizar_lista(slopemin_lpv))
slopemin_No=np.array(estandarizar_lista(slopemin_No))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(slopemin_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(slopemin_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(slopemin_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(slopemin_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(slopemin_No, bins=10,  alpha=0.5, density=True, label='None') 
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('Slope_min')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('CSlope_min.png')
# Mostrar la gráfica
plt.show()
plt.close()
print("calcule sople min")



# 18. r_value

def calcular_r_value(s_variables, HJD):
    # Lista para almacenar los resultados de la característica "r_value"
    resultados_r_value = []
    
    for sublist in s_variables:
        # Inicializar la lista para almacenar los valores de r_value de las regresiones lineales
        r_values = []
        
        # Calcular la regresión lineal para cada subintervalo
        for i in range(len(HJD) - 1):
            # Seleccionar los datos dentro del intervalo de tiempo
            x = []
            y = []
            for j in range(len(HJD)):
                if HJD[i] <= HJD[j] <= HJD[i + 1]:
                    x.append(HJD[j])
                    y.append(sublist[j])
            
            # Calcular la regresión lineal para el subintervalo
            if len(x) > 1:
                _, _, r_value, _, _ = linregress(x, y)
                r_values.append(r_value)
        
        # Calcular el promedio de los valores de r_value de las regresiones lineales
        if r_values:
            r_value_avg = sum(r_values) / len(r_values)
        else:
            r_value_avg = None
        
        # Agregar el resultado a la lista de resultados
        resultados_r_value.append(r_value_avg)
    
    return resultados_r_value


# Uso de la función con s_variables y HJD, ignorando s_novariables
r_value_cefeidas= calcular_r_value(cefeidas, HJD)
r_value_binarias=calcular_r_value(binarias, HJD)
r_value_lyrae= calcular_r_value(lyrae, HJD)
r_value_lpv=calcular_r_value(lpv, HJD)
r_value_No=calcular_r_value(No, HJD)
#Estandarizar
r_value_cefeidas=np.array(estandarizar_lista(r_value_cefeidas))
r_value_binarias=np.array(estandarizar_lista(r_value_binarias))
r_value_lyrae=np.array(estandarizar_lista(r_value_lyrae))
r_value_lpv=np.array(estandarizar_lista(r_value_lpv))
r_value_No=np.array(estandarizar_lista(r_value_No))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(r_value_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(r_value_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(r_value_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(r_value_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(r_value_No, bins=10,  alpha=0.5, density=True, label='None') 
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('r_value')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('Cr_value.png')
# Mostrar la gráfica
plt.show()
plt.close()
print("calcule r value")




# 19. r_val_min

def calcular_r_value_min(s_variables, HJD):
    # Lista para almacenar los resultados de la característica "r_value_min"
    resultados_r_value_min = []
    
    for sublist in s_variables:
        # Inicializar la lista para almacenar los valores de r_value de las regresiones lineales
        r_values = []
        
        # Calcular la regresión lineal para cada subintervalo
        for i in range(len(HJD) - 1):
            # Seleccionar los datos dentro del intervalo de tiempo
            x = []
            y = []
            for j in range(len(HJD)):
                if HJD[i] <= HJD[j] <= HJD[i + 1]:
                    x.append(HJD[j])
                    y.append(sublist[j])
            
            # Calcular la regresión lineal para el subintervalo
            if len(x) > 1:
                _, _, r_value, _, _ = linregress(x, y)
                r_values.append(r_value)
        
        # Calcular el mínimo de los valores de r_value de las regresiones lineales
        if r_values:
            r_value_min = min(r_values)
        else:
            r_value_min = None
        
        # Agregar el resultado a la lista de resultados
        resultados_r_value_min.append(r_value_min)
    
    return resultados_r_value_min


# Uso de la función con s_variables y HJD, ignorando s_novariables
r_valuemin_cefeidas= calcular_r_value_min(cefeidas, HJD)
r_valuemin_binarias=calcular_r_value_min(binarias, HJD)
r_valuemin_lyrae= calcular_r_value_min(lyrae, HJD)
r_valuemin_lpv=calcular_r_value_min(lpv, HJD)
r_valuemin_No=calcular_r_value_min(No, HJD)
#Estandarizar
r_valuemin_cefeidas=np.array(estandarizar_lista(r_valuemin_cefeidas))
r_valuemin_binarias=np.array(estandarizar_lista(r_valuemin_binarias))
r_valuemin_lyrae=np.array(estandarizar_lista(r_valuemin_lyrae))
r_valuemin_lpv=np.array(estandarizar_lista(r_valuemin_lpv))
r_valuemin_No=np.array(estandarizar_lista(r_valuemin_No))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(r_valuemin_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(r_valuemin_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(r_valuemin_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(r_valuemin_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(r_valuemin_No, bins=10,  alpha=0.5, density=True, label='None') 

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('r_value_min')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('Cr_value_min.png')
# Mostrar la gráfica
plt.show()
plt.close()
print("calcule r_min")



# 20. eta
def calcular_eta(s_variables, HJD):
    # Lista para almacenar los resultados de la característica "eta"
    resultados_eta = []
    
    for sublist in s_variables:
        # Calcular el numerador de la característica "eta"
        numerador = sum([(y2 - y1)**2 for y1, y2 in zip(sublist[:-1], sublist[1:])])
        
        # Calcular el denominador de la característica "eta"
        denominador = sum([(y - np.mean(sublist))**2 for y in sublist])
        
        # Calcular la característica "eta"
        if denominador != 0:
            eta = numerador / denominador
        else:
            eta = None
        
        # Agregar el resultado a la lista de resultados
        resultados_eta.append(eta)
    
    return resultados_eta


eta_cefeidas= calcular_eta(cefeidas, HJD)
eta_binarias=calcular_eta(binarias, HJD)
eta_lyrae= calcular_eta(lyrae, HJD)
eta_lpv=calcular_eta(lpv, HJD)
eta_No=calcular_eta(No, HJD)
#Estandarizar
eta_cefeidas=np.array(estandarizar_lista(eta_cefeidas))
eta_binarias=np.array(estandarizar_lista(eta_binarias))
eta_lyrae=np.array(estandarizar_lista(eta_lyrae))
eta_lpv=np.array(estandarizar_lista(eta_lpv))
eta_No=np.array(estandarizar_lista(eta_No))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(eta_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(eta_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(eta_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(eta_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(eta_No, bins=10,  alpha=0.5, density=True, label='None')
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('eta')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('etaC.png')
# Mostrar la gráfica
plt.show()
plt.close()
print("calcular eta")



# 21. reDSign

def calcular_reDSign(s_variables, HJD):
    # Lista para almacenar los resultados de la característica "reDSign"
    resultados_reDSign = []
    
    for sublist in s_variables:
        # Calcular las diferencias entre puntos consecutivos
        diffs = [y2 - y1 for y1, y2 in zip(sublist[:-1], sublist[1:])]
        
        # Calcular las diferencias signadas
        signos = [np.sign(diff) for diff in diffs]
        
        # Calcular el producto de las diferencias signadas y el signo de la pendiente
        reDSign = np.median([signo * np.sign(sublist[-1] - sublist[0]) for signo in signos])
        
        # Agregar el resultado a la lista de resultados
        resultados_reDSign.append(reDSign)
    
    return resultados_reDSign


reDSign_cefeidas= calcular_reDSign(cefeidas, HJD)
reDSign_binarias=calcular_reDSign(binarias, HJD)
reDSign_lyrae= calcular_reDSign(lyrae, HJD)
reDSign_lpv=calcular_reDSign(lpv, HJD)
reDSign_No=calcular_reDSign(No, HJD)
#Estandarizar
reDSign_cefeidas=np.array(estandarizar_lista(reDSign_cefeidas))
reDSign_binarias=np.array(estandarizar_lista(reDSign_binarias))
reDSign_lyrae=np.array(estandarizar_lista(reDSign_lyrae))
reDSign_lpv=np.array(estandarizar_lista(reDSign_lpv))
reDSign_No=np.array(estandarizar_lista(reDSign_No))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(reDSign_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(reDSign_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(reDSign_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(reDSign_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(reDSign_No, bins=10,  alpha=0.5, density=True, label='None')

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('reDSign')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('CreDSingn.png')
# Mostrar la gráfica
plt.show()
plt.close()
print("calcule Rsing")



# 22. rbLeon

def calcular_rbLeon(s_variables, HJD):
    # Lista para almacenar los resultados de la característica "rbLeon"
    resultados_rbLeon = []
    
    for sublist in s_variables:
        # Calcular las diferencias entre puntos consecutivos
        diffs = [y2 - y1 for y1, y2 in zip(sublist[:-1], sublist[1:])]
        
        # Calcular la mediana de las diferencias absolutas
        rbLeon = np.median(np.abs(diffs))
        
        # Agregar el resultado a la lista de resultados
        resultados_rbLeon.append(rbLeon)
    
    return resultados_rbLeon



rbLeon_cefeidas= calcular_rbLeon(cefeidas, HJD)
rbLeon_binarias=calcular_rbLeon(binarias, HJD)
rbLeon_lyrae= calcular_rbLeon(lyrae, HJD)
rbLeon_lpv=calcular_rbLeon(lpv, HJD)
rbLeon_No=calcular_rbLeon(No, HJD)
#Estandarizar
rbLeon_cefeidas=np.array(estandarizar_lista(rbLeon_cefeidas))
rbLeon_binarias=np.array(estandarizar_lista(rbLeon_binarias))
rbLeon_lyrae=np.array(estandarizar_lista(rbLeon_lyrae))
rbLeon_lpv=np.array(estandarizar_lista(rbLeon_lpv))
rbLeon_No=np.array(estandarizar_lista(rbLeon_No))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(rbLeon_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(rbLeon_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(rbLeon_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(rbLeon_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(rbLeon_No, bins=10,  alpha=0.5, density=True, label='None')

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('rbLeon')
ax.set_xlabel('Values')
ax.set_ylabel('Density')

# Ajustar el diseño
plt.tight_layout()

# Guardar el gráfico en un archivo
plt.savefig('CrbLeon.png')

# Mostrar la gráfica
plt.show()
plt.close()

print("rbLeon")

# 23. rbLeonsign

def calcular_rbLeon_sign(s_variables):
    # Lista para almacenar los resultados de la característica "rbLeon_sign"
    resultados_rbLeon_sign = []
    
    for sublist in s_variables:
        # Calcular las diferencias entre puntos consecutivos
        diffs = [y2 - y1 for y1, y2 in zip(sublist[:-1], sublist[1:])]
        
        # Calcular el signo de las diferencias
        signos_diffs = [np.sign(diff) for diff in diffs]
        
        # Calcular la mediana de los signos de las diferencias
        rbLeon_sign = np.median(signos_diffs)
        
        # Agregar el resultado a la lista de resultados
        resultados_rbLeon_sign.append(rbLeon_sign)
    
    return resultados_rbLeon_sign

srbLeon_cefeidas= calcular_rbLeon_sign(cefeidas)
srbLeon_binarias=calcular_rbLeon_sign(binarias)
srbLeon_lyrae= calcular_rbLeon_sign(lyrae)
srbLeon_lpv=calcular_rbLeon_sign(lpv)
srbLeon_No=calcular_rbLeon_sign(No)
#Estandarizar
srbLeon_cefeidas=np.array(estandarizar_lista(srbLeon_cefeidas))
srbLeon_binarias=np.array(estandarizar_lista(srbLeon_binarias))
srbLeon_lyrae=np.array(estandarizar_lista(srbLeon_lyrae))
srbLeon_lpv=np.array(estandarizar_lista(srbLeon_lpv))
srbLeon_No=np.array(estandarizar_lista(srbLeon_No))
# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(srbLeon_cefeidas, bins=10,  alpha=0.5, density=True, label='Cefeidas')
ax.hist(srbLeon_binarias, bins=10,  alpha=0.5, density=True, label='Binarias')
ax.hist(srbLeon_lyrae, bins=10,  alpha=0.5, density=True, label='Lyrae')
ax.hist(srbLeon_lpv, bins=10,  alpha=0.5, density=True, label='LPV')
ax.hist(srbLeon_No, bins=10,  alpha=0.5, density=True, label='None')

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('rbLeon_sign')
ax.set_xlabel('Values')
ax.set_ylabel('Density')

# Ajustar el diseño
plt.tight_layout()

# Guardar el gráfico en un archivo
plt.savefig('CrbLeon_singn.png')

# Mostrar la gráfica
plt.show()
plt.close()
print("rbleon sing")


# In[ ]:


# Establecer la semilla
random.seed(42)  # Puedes cambiar este número por cualquier otro valor entero
# Generar números aleatorios
aleatorio = [random.random() for _ in range(608)]
aleatorio = np.array(aleatorio)

# Concatenar todas las variables de forma independiente
mads = np.concatenate([mad_cefeidas, mad_binarias, mad_lyrae, mad_lpv, mad_no])
print(len(mads))
mean = np.concatenate([mean_cefeidas, mean_binarias, mean_lyrae, mean_lpv, mean_no])
print(len(mean))
VART_s = np.concatenate([VART_cefeidas, VART_binarias, VART_lyrae, VART_lpv, VART_No])
print(len(VART_s))
av_s = np.concatenate([av_cefeidas, av_binarias, av_lyrae, av_lpv, av_No])
print(len(av_s))
rob_s = np.concatenate([rob_cefeidas, rob_binarias, rob_lyrae, rob_lpv, rob_No])
print(len(rob_s))
DIFDER_s = np.concatenate([DIFDER_cefeidas, DIFDER_binarias, DIFDER_lyrae, DIFDER_lpv, DIFDER_No])
print(len(DIFDER_s))
proy = np.concatenate([PROY_cefeidas, PROY_binarias, PROY_lyrae, PROY_lpv, PROY_No])
print(len(proy))
integrals = np.concatenate([integrals_cefeidas, integrals_binarias, integrals_lyrae, integrals_lpv, integrals_No])
print(len(integrals))
rulds = np.concatenate([RULD_cefeidas, RULD_binarias, RULD_lyrae, RULD_lpv, RULD_No])
print(len(rulds))
os = np.concatenate([os_cefeidas, os_binarias, os_lyrae, os_lpv, os_No])
print(len(os))
rEucliDs = np.concatenate([rEucliDs_cefeidas, rEucliDs_binarias, rEucliDs_lyrae, rEucliDs_lpv, rEucliDs_No])
print(len(rEucliDs))
resultados_slope_min = np.concatenate([slopemin_cefeidas, slopemin_binarias, slopemin_lyrae, slopemin_lpv, slopemin_No])
print(len(resultados_slope_min))
resultados_low = np.concatenate([low_cefeidas, low_binarias, low_lyrae, low_lpv, low_No])
print(len(resultados_low))
resultados_row = np.concatenate([row_cefeidas, row_binarias, row_lyrae, row_lpv, row_No])
print(len(resultados_row))
resultados_DeltaM = np.concatenate([DeltaM_cefeidas, DeltaM_binarias, DeltaM_lyrae, DeltaM_lpv, DeltaM_No])
print(len(resultados_DeltaM))
resultados_slope = np.concatenate([slope_cefeidas, slope_binarias, slope_lyrae, slope_lpv, slope_No])
print(len(resultados_slope))
resultados_r_value = np.concatenate([r_value_cefeidas, r_value_binarias, r_value_lyrae, r_value_lpv, r_value_No])
print(len(resultados_r_value))
resultados_r_value_min = np.concatenate([r_valuemin_cefeidas, r_valuemin_binarias, r_valuemin_lyrae, r_valuemin_lpv, r_valuemin_No])
print(len(resultados_r_value_min))
resultados_eta = np.concatenate([eta_cefeidas, eta_binarias, eta_lyrae, eta_lpv, eta_No])
print(len(resultados_eta))
resultados_reDSign = np.concatenate([reDSign_cefeidas, reDSign_binarias, reDSign_lyrae, reDSign_lpv, reDSign_No])
print(len(resultados_reDSign))
resultados_rbLeon = np.concatenate([rbLeon_cefeidas, rbLeon_binarias, rbLeon_lyrae, rbLeon_lpv, rbLeon_No])
print(len(resultados_rbLeon))
resultados_rbLeon_sign = np.concatenate([srbLeon_cefeidas, srbLeon_binarias, srbLeon_lyrae, srbLeon_lpv, srbLeon_No])
print(len(resultados_rbLeon_sign))
print('Concatenar')


h2o.init()
STYPE=[]
for i in range(len(cefeidas)):
    STYPE.append('cefeidas')
for j in range(len(binarias)):
    STYPE.append('binarias')
for m in range(len(lyrae)):
    STYPE.append('lyrae')
for n in range(len(lpv)):
    STYPE.append('lpv')
for x in range (len(No)): 
    STYPE.append('None')
STYPE= np.array(STYPE)
print('Se han transcrito las variables a formato H2O')
print(len(STYPE))
print(len(aleatorio))

#Creamos el H2O frame con una columna
datos = h2o.H2OFrame(python_obj=STYPE, column_names=['TYPE'], column_types=["string"])
print(len(datos))

# Variables que quieres agregar
variables = [mads ,  mean , VART_s,  av_s ,  rob_s ,  DIFDER_s ,  proy,  integrals,  rulds,  os ,  rEucliDs, 
resultados_slope_min, resultados_low, resultados_row,  resultados_DeltaM,  resultados_slope, resultados_r_value, resultados_r_value_min, resultados_eta, resultados_reDSign, resultados_rbLeon, resultados_rbLeon_sign, aleatorio]

print(len(variables))

# Nombres de las columnas correspondientes
nombres_columnas = ['mads' ,  'mean' , 'VART_s',  'av_s' ,  'rob_s' ,  'DIFDER_s' ,  'proy',  'integrals',  'rulds',  'os' ,  'rEucliDs', 
'resultados_slope_min', 'resultados_low', 'resultados_row',  'resultados_DeltaM',  'resultados_slope', 'resultados_r_value', 'resultados_r_value_min', 'resultados_eta', 'resultados_reDSign', 'resultados_rbLeon', 'resultados_rbLeon_sign', 'aleatorio']
print(len(nombres_columnas))

# Crear y agregar cada columna al marco de datos 'datos'

for variable, nombre_columna in zip(variables, nombres_columnas):
    # Convertir la variable a un marco de datos H2O
    variable_h2o = h2o.H2OFrame(python_obj=variable, column_names=[nombre_columna], column_types=["float"])
    # Agregar la columna al marco de datos existente
    datos = datos.cbind(variable_h2o)

# In[ ]:

print("se creo el frame")

# Obtener el nombre de todas las columnas excepto la primera
columnas_numericas = datos.columns[1:]  # Excluir la primera columna 'SPECTYPE'
# Seleccionar solo las columnas numéricas
datos_numericos = datos[columnas_numericas]
# Calcular la matriz de correlación
correlation_matrix = datos_numericos.cor().as_data_frame()
correlation_matrix.to_csv('Catriz_correlacion.csv', index=False)
print('se ha calculado la matriz de correlacion')


# Crear una figura
plt.figure(figsize=(10, 8))
# Generar un mapa de calor con la matriz de correlación
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# Ajustar la disposición para que no se corte nada
plt.tight_layout()
# Guardar la figura como un archivo PNG
plt.savefig('Cmatriz_correlacion.png')
# Mostrar la figura
plt.show()
plt.close()


# Convierte la variable objetivo a factor utilizando la función asfactor()
datos['TYPE'] = datos['TYPE'].asfactor()

# Define las columnas predictoras y la variable objetivo
predictores = datos.columns[1:]  # Todas las columnas excepto la primera (SPECTYPE)
objetivo='TYPE'
train, test = datos.split_frame(ratios=[0.6], seed=42)
# Configura y entrena el modelo de Random Forest
modelo_rf = H2ORandomForestEstimator(ntrees=200, max_depth=20, seed=42)
modelo_rf.train(x=predictores, y=objetivo, training_frame=train, validation_frame=test)

# Imprime métricas de rendimiento en el conjunto de prueba
print(modelo_rf.model_performance(test_data=test))

# Obtener las importancias de las variables
importancias_variables_rf = modelo_rf.varimp(True)
print("\nImportancias relativas de las características de entrenamiento:")
print(importancias_variables_rf)
# In[ ]:

print('termine features')


#Random Forest
STYPE=[]
for i in range(len(cefeidas)):
    STYPE.append('cefeidas')
for j in range(len(binarias)):
    STYPE.append('binarias')
for m in range(len(lyrae)):
    STYPE.append('lyrae')
for n in range(len(lpv)):
    STYPE.append('lpv')
for x in range (len(No)): 
    STYPE.append('None')
STYPE= np.array(STYPE)
print('Se han transcrito las variables a formato H2O')


#Creamos el H2O frame con una columna
datos1 = h2o.H2OFrame(python_obj=STYPE, column_names=['TYPE'], column_types=["string"])
print(len(datos))

# Variables que quieres agregar
variables1 = [mean , proy,  integrals, rEucliDs]
print(type(variables1))

# Nombres de las columnas correspondientes
nombres_columnas1 = ['mean' ,  'proy',  'integrals', 'rEucliDs']
print(type(nombres_columnas1))

# Crear y agregar cada columna al marco de datos 'datos'

for variable, nombre_columna in zip(variables1, nombres_columnas1):
    # Convertir la variable a un marco de datos H2O
    variable1_h2o = h2o.H2OFrame(python_obj=variable, column_names=[nombre_columna], column_types=["float"])
    # Agregar la columna al marco de datos existente
    datos1 = datos1.cbind(variable1_h2o)
print("se creo el frame")



# Convertir la columna 'TYPE' en categórica
datos1['TYPE'] = datos1['TYPE'].asfactor()

# Dividir los datos en entrenamiento y prueba
train, test = datos1.split_frame(ratios=[0.8])

# Lista de características y variable objetivo
features = ['integrals', 'mean', 'proy', 'rEucliDs']
target = 'TYPE'

# Entrenar el modelo RandomForest
rf_model = H2ORandomForestEstimator(ntrees=300, max_depth=100, nfolds=5)
rf_model.train(x=features, y=target, training_frame=train)

# Evaluar el modelo en el conjunto de prueba

performance = rf_model.model_performance(test_data=test)

# Obtener la matriz de confusión
confusion_matrix = performance.confusion_matrix()

print(confusion_matrix)



h2o.cluster().shutdown()
print('Proceso finalizado')
