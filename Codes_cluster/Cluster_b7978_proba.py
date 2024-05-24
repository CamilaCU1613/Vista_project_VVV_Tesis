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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
import joblib
print('importe las liberias')


# In[2]:


archivo = fits.open('Features')
s_variables78= archivo[1].data
s_novariables78= archivo[2].data
HJD78= archivo[3].data
s_variables79= archivo[6].data
s_novariables79= archivo[4].data
HJD79= archivo[5].data
archivo.info()


# In[9]:


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


# In[12]:


#  1. MAD
# Función para calcular la Mediana de la Desviación Absoluta (MAD) eliminando valores np.nan
def mad(arr1, arr2):
    mads_arr1 = []
    for sublist in arr1:
        arr_cleaned = [x for x in sublist if not np.isnan(x)]
        if arr_cleaned:  # Verifica que la lista no esté vacía después de eliminar np.nan
            median = np.median(arr_cleaned)
            mad = np.median(np.abs(arr_cleaned - median))
            mads_arr1.append(mad)
    
    mads_arr2 = []
    for sublist in arr2:
        arr_cleaned = [x for x in sublist if not np.isnan(x)]
        if arr_cleaned:  # Verifica que la lista no esté vacía después de eliminar np.nan
            median = np.median(arr_cleaned)
            mad = np.median(np.abs(arr_cleaned - median))
            mads_arr2.append(mad)
    
    return mads_arr1, mads_arr2
# Calcular MAD para los datos
mads_s_variables78, mads_s_novariables78 = mad(s_variables78, s_novariables78)
mads_s_variables79, mads_s_novariables79 = mad(s_variables79, s_novariables79)
mads_s_variables= mads_s_variables78+mads_s_variables79
mads_s_novariables=mads_s_novariables78+mads_s_novariables79
mads_s_variables, mads_s_novariables = estandarizar_lista(mads_s_variables), estandarizar_lista(mads_s_novariables )
print(len(mads_s_variables))
print(len(mads_s_novariables))


# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(mads_s_variables, bins=10, color='blue', alpha=0.5, density=True, label='s_variables')
ax.hist(mads_s_novariables, bins=10, color='green', alpha=0.5, density=True, label='s_novariables')
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('MAD')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('C_MAD.png')
# Mostrar la gráfica
plt.show()
plt.close()

# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(mads_s_variables)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(mads_s_novariables < threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine MAD")


# In[ ]:


# 2. Mediana

def calcular_mediana_por_objeto(s_variables, s_novariables):
   # Lista para almacenar los resultados
   resultados_variables = []
   resultados_novariables = []
   # Iterar sobre las sub listas en s_variables
   for sublist in s_variables:
       # Filtrar np.nan de la sub lista
       sublist_filtrada = [x for x in sublist if not np.isnan(x)]
       # Calcular la mediana
       mediana = np.median(sublist_filtrada)
       # Agregar la mediana a los resultados
       resultados_variables.append(mediana)
   # Iterar sobre las sub listas en s_novariables
   for sublist in s_novariables:
       # Filtrar np.nan de la sub lista
       sublist_filtrada = [x for x in sublist if not np.isnan(x)]
       # Calcular la mediana
       mediana = np.median(sublist_filtrada)
       # Agregar la mediana a los resultados
       resultados_novariables.append(mediana)

   return resultados_variables, resultados_novariables

resultados_variables78, resultados_novariables78 = calcular_mediana_por_objeto(s_variables78, s_novariables78)
resultados_variables79, resultados_novariables79 = calcular_mediana_por_objeto(s_variables79, s_novariables79)
resultados_variables=resultados_variables78+resultados_variables79
resultados_novariables=resultados_novariables78 +resultados_novariables79
resultados_variables, resultados_novariables = estandarizar_lista(resultados_variables), estandarizar_lista(resultados_novariables)
print(len(resultados_variables))
print(len(resultados_novariables))


# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(resultados_variables, bins=10, color='blue', alpha=0.5, density=True, label='s_variables')
ax.hist(resultados_novariables, bins=10, color='green', alpha=0.5, density=True, label='s_novariables')
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('Mean')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('C_Mediana.png')
# Mostrar la gráfica
plt.show()
plt.close()
# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(resultados_variables)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(resultados_novariables < threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine Mean")


# In[ ]:


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
VART_s_variables78 = [calcular_VART(sublista, HJD78) for sublista in s_variables78]
VART_s_novariables78 = [calcular_VART(sublista, HJD78) for sublista in s_novariables78]
VART_s_variables79 = [calcular_VART(sublista, HJD79) for sublista in s_variables79]
VART_s_novariables79 = [calcular_VART(sublista, HJD79) for sublista in s_novariables79]
VART_s_variables=VART_s_variables78+VART_s_variables79
VART_s_novariables= VART_s_novariables78+VART_s_novariables79
VART_s_variables, VART_s_novariables = estandarizar_lista(VART_s_variables), estandarizar_lista(VART_s_novariables)
print(len(VART_s_variables))
print(len(VART_s_novariables))

# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(VART_s_variables, bins=10, color='blue', alpha=0.5, density=True, label='s_variables')
ax.hist(VART_s_novariables, bins=10, color='green', alpha=0.5, density=True, label='s_novariables')
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('VART')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('c_VART.png')
# Mostrar la gráfica
plt.show()
plt.close()

# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(VART_s_variables)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(VART_s_novariables < threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine VART")


# In[ ]:


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

av_s_variables78 = calcular_av_para_listas_de_listas(s_variables78)
av_s_novariables78 = calcular_av_para_listas_de_listas(s_novariables78)
av_s_variables79 = calcular_av_para_listas_de_listas(s_variables79)
av_s_novariables79 = calcular_av_para_listas_de_listas(s_novariables79)
av_s_variables= av_s_variables78+av_s_variables79
av_s_novariables=av_s_novariables78+av_s_novariables79
av_s_variables, av_s_novariables = estandarizar_lista(av_s_variables), estandarizar_lista(av_s_novariables)
print(len(av_s_variables))
print(len(av_s_novariables))


# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(av_s_variables, bins=10, color='blue', alpha=0.5, density=True, label='s_variables')
ax.hist(av_s_novariables, bins=10, color='green', alpha=0.5, density=True, label='s_novariables')
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('AV Factor')
ax.set_xlabel('Values')
ax.set_ylabel('Density')

# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('C_AV.png')
# Mostrar la gráfica
plt.show()
plt.close()

# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(av_s_variables)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(av_s_novariables < threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine AV")


# In[ ]:


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

def calcular_robAbbe_separado(s_variables, s_novariables, c=1.345):
    resultados_s_variables = []
    resultados_s_novariables = []

    for sublist_vars in s_variables:
        robAbbe_vars = calcular_robAbbe(sublist_vars, c=c)
        resultados_s_variables.append(np.nanmean(robAbbe_vars))  # Agregar la media de los resultados

    for sublist_novars in s_novariables:
        robAbbe_novars = calcular_robAbbe(sublist_novars, c=c)
        resultados_s_novariables.append(np.nanmean(robAbbe_novars))  # Agregar la media de los resultados

    return resultados_s_variables, resultados_s_novariables

rob_s_variables78, rob_s_novariables78 = calcular_robAbbe_separado(s_variables78, s_novariables78)
rob_s_variables79, rob_s_novariables79 = calcular_robAbbe_separado(s_variables79, s_novariables79)
rob_s_variables=rob_s_variables78+rob_s_variables79
rob_s_novariables=rob_s_novariables78+rob_s_novariables79
print(len(rob_s_variables))
print(len(rob_s_novariables))

# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))

# Graficar histogramas en el mismo gráfico
ax.hist(rob_s_variables, bins=10, color='blue', alpha=0.5, density=True, label='s_variables')
ax.hist(rob_s_novariables, bins=10, color='green', alpha=0.5, density=True, label='s_novariables')

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('robAbbe')
ax.set_xlabel('Values')
ax.set_ylabel('Density')

# Ajustar el diseño
plt.tight_layout()

# Guardar el gráfico en un archivo
plt.savefig('C_robAbbe.png')

# Mostrar la gráfica
plt.show()
plt.close()

# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(rob_s_variables)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(rob_s_novariables < threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine robAbbe")


# In[ ]:


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


DIFDER_s_variables78 = calcular_DIFDER(HJD78, s_variables78)
DIFDER_s_variables79 = calcular_DIFDER(HJD79, s_variables79)
DIFDER_s_novariables78 = calcular_DIFDER(HJD78, s_novariables78)
DIFDER_s_novariables79 = calcular_DIFDER(HJD79, s_novariables79)
DIFDER_s_variables=DIFDER_s_variables78+DIFDER_s_variables79
DIFDER_s_novariables=DIFDER_s_novariables78 +DIFDER_s_novariables79
DIFDER_s_variables, DIFDER_s_novariables = estandarizar_lista(DIFDER_s_variables), estandarizar_lista(DIFDER_s_novariables)
print(len(DIFDER_s_variables))
print(len(DIFDER_s_novariables))

# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(DIFDER_s_variables, bins=10, color='blue', alpha=0.5, density=True, label='s_variables')
ax.hist(DIFDER_s_novariables, bins=10, color='green', alpha=0.5, density=True, label='s_novariables')
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('DIFDER')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('C_DIFDE.png')
# Mostrar la gráfica
plt.show()
plt.close()
# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(DIFDER_s_variables)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(DIFDER_s_novariables < threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine DIFDER")


# In[ ]:


# 7. PROY 2

# Función para eliminar los valores nan de una lista
def remove_nan(lst):
    return [x for x in lst if not np.isnan(x)]

# Eliminar los valores nan de las listas en s_variables y guardar en HDJ_variables
variables78 = [remove_nan(lst) for lst in s_variables78]
variables79 = [remove_nan(lst) for lst in s_variables79]
# No eliminar los valores nan de las listas en s_novariables y guardar en HJD_novariables
novariables78 = [remove_nan(lst) for lst in s_novariables78]
novariables79 = [remove_nan(lst) for lst in s_novariables79]

# Función para eliminar np.nan de HJD por cada sublista de s_variables
def remove_nan_in_HJD(HJD, s_variables):
    result = []
    for sublist in s_variables:
        valid_indices = [i for i, val in enumerate(sublist) if not np.isnan(val)]
        result.append([HJD[i] for i in valid_indices])
    return result

# Eliminar np.nan en HJD por cada sublista de s_variables
HJD_variable78 = remove_nan_in_HJD(HJD78, s_variables78)
HJD_novariable78 = remove_nan_in_HJD(HJD78, s_novariables78)
HJD_variable79 = remove_nan_in_HJD(HJD79, s_variables79)
HJD_novariable79 = remove_nan_in_HJD(HJD79, s_novariables79)

# Función para calcular la proyección según la fórmula dada
def calcular_proyeccion(X, Y):
    N = len(X)
    proyecciones = []
    for i in range(1, N-1):
        proyeccion = ((Y[i+1] - Y[i-1]) / (X[i+1] - X[i-1])) * (X[i] - X[i-1]) + Y[i-1] - Y[i]
        proyecciones.append(proyeccion)
    return proyecciones

# Calcular proyección para cada sublista y guardar los resultados en una lista
proy_varibles78 = []
for i in range(len(HJD_variable78)):
    X_actual = HJD_variable78[i]
    Y_actual = variables78[i]
    proyecciones_actual = calcular_proyeccion(X_actual, Y_actual)
    proyeccion_promedio = sum(proyecciones_actual) / len(proyecciones_actual)
    proy_varibles78.append(proyeccion_promedio)
proy_varibles79 = []
for i in range(len(HJD_variable79)):
    X_actual = HJD_variable79[i]
    Y_actual = variables79[i]
    proyecciones_actual = calcular_proyeccion(X_actual, Y_actual)
    proyeccion_promedio = sum(proyecciones_actual) / len(proyecciones_actual)
    proy_varibles79.append(proyeccion_promedio)
    
    
# Calcular proyección para cada sublista y guardar los resultados en una lista
proy_novaribles78 = []
for j in range(len(HJD_novariable78)):
    X_actual = HJD_novariable78[j]
    Y_actual = novariables78[j]
    proyecciones_actual = calcular_proyeccion(X_actual, Y_actual)
    proyeccion_promedio = sum(proyecciones_actual) / len(proyecciones_actual)
    proy_novaribles78.append(proyeccion_promedio)
 # Calcular proyección para cada sublista y guardar los resultados en una lista
proy_novaribles79 = []
for j in range(len(HJD_novariable79)):
    X_actual = HJD_novariable79[j]
    Y_actual = novariables79[j]
    proyecciones_actual = calcular_proyeccion(X_actual, Y_actual)
    proyeccion_promedio = sum(proyecciones_actual) / len(proyecciones_actual)
    proy_novaribles79.append(proyeccion_promedio)
    
proy_varibles= proy_varibles78+ proy_varibles79
proy_novaribles=proy_novaribles78+proy_novaribles79
proy_varibles, proy_novaribles = estandarizar_lista(proy_varibles), estandarizar_lista(proy_novaribles)

print(len(proy_varibles))
print(len(proy_novaribles))


# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico con rango en el eje x de -20 a 20
ax.hist(proy_varibles, bins=10, color='blue', alpha=0.5, density=True, label='s_variables', range=(-20, 20))
ax.hist(proy_novaribles, bins=10, color='green', alpha=0.5, density=True, label='s_novariables', range=(-20, 20))
# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('PROY 2')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
# Ajustar el diseño
plt.tight_layout()

# Guardar el gráfico en un archivo
plt.savefig('C_PROY2.png')
# Mostrar la gráfica
plt.show()
plt.close()

# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(proy_varibles)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(proy_novaribles < threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine Proy2")


# In[ ]:


# 8. INTEGRAL 
def calcular_integral(X, Y):
    N = len(X)
    delta_X = [X[i + 1] - X[i] for i in range(N - 1)]
    mean_Y = sum(Y) / N
    integral = sum([(delta_X[i] * (Y[i] - mean_Y)) for i in range(N - 1)]) / (X[N - 1] - X[0])
    return integral

integrals_variables78 = []
for i in range(len(HJD_variable78)):
    integral = calcular_integral(HJD_variable78[i], variables78[i])
    integrals_variables78.append(abs(integral))

integrals_variables79 = []
for i in range(len(HJD_variable79)):
    integral = calcular_integral(HJD_variable79[i], variables79[i])
    integrals_variables79.append(abs(integral))
    
    
integrals_novariables78 = []
for j in range(len(HJD_novariable78)):
    integral = calcular_integral(HJD_novariable78[j], novariables78[j])
    integrals_novariables78.append(abs(integral))

integrals_novariables79 = []
for j in range(len(HJD_novariable79)):
    integral = calcular_integral(HJD_novariable79[j], novariables79[j])
    integrals_novariables79.append(abs(integral))

integrals_variables=integrals_variables78+integrals_variables79
integrals_novariables=integrals_novariables78+integrals_novariables79
integrals_variables, integrals_novariables = estandarizar_lista(integrals_variables), estandarizar_lista(integrals_novariables)
print(len(integrals_variables))
print(len(integrals_novariables))


# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))

# Graficar histogramas en el mismo gráfico
ax.hist(integrals_variables, bins=10, color='blue', alpha=0.5, density=True, label='s_variables', range= (0,25))
ax.hist(integrals_novariables, bins=10, color='green', alpha=0.5, density=True, label='s_novariables', range= (0,25))

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('INTEGRAL')
ax.set_xlabel('Values')
ax.set_ylabel('Density')

# Ajustar el diseño
plt.tight_layout()

# Guardar el gráfico en un archivo
plt.savefig('C_Integral.png')

# Mostrar la gráfica
plt.show()
plt.close()

# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(integrals_variables)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(integrals_novariables < threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine integral")


# In[ ]:


# 10. RULD

def calcular_RULD(X, Y):
    N = len(X)
    ruld_sum = sum([((Y[i] - Y[i + 1]) / abs(Y[i] - Y[i + 1])) * (X[i + 1] - X[i]) for i in range(N - 1)])
    ruld = (1 / (N - 1)) * ruld_sum
    return ruld

rulds_variables78 = []
for i in range(len(HJD_variable78)):
    ruld = calcular_RULD(HJD_variable78[i], variables78[i])
    rulds_variables78.append(ruld)
rulds_variables79 = []
for i in range(len(HJD_variable79)):
    ruld = calcular_RULD(HJD_variable79[i], variables79[i])
    rulds_variables79.append(ruld)
    
rulds_novariables78 = []
for j in range(len(HJD_novariable78)):
    ruld = calcular_RULD(HJD_novariable78[j], novariables78[j])
    rulds_novariables78.append(ruld)
rulds_novariables79 = []
for j in range(len(HJD_novariable79)):
    ruld = calcular_RULD(HJD_novariable79[j], novariables79[j])
    rulds_novariables79.append(ruld)
    
rulds_variables=rulds_variables78+rulds_variables79
rulds_novariables=rulds_novariables78+rulds_novariables79
rulds_variables, rulds_novariables = estandarizar_lista(rulds_variables), estandarizar_lista(rulds_novariables)   
print(len(rulds_variables))
print(len(rulds_novariables))


# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))
# Graficar histogramas en el mismo gráfico
ax.hist(rulds_variables, bins=10, color='blue', alpha=0.5, density=True, label='s_variables')
ax.hist(rulds_novariables, bins=10, color='green', alpha=0.5, density=True, label='s_novariables')

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('RULD')
ax.set_xlabel('Values')
ax.set_ylabel('Density')

# Ajustar el diseño
plt.tight_layout()

# Guardar el gráfico en un archivo
plt.savefig('C_RULD.png')

# Mostrar la gráfica
plt.show()
plt.close()

# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(rulds_variables)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(rulds_novariables< threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine RULD")


# In[ ]:


# 11. Asimetria de octil (OS)

# Función para calcular la asimetría de octil (OS)
def calcular_os(lista):
    q1 = np.percentile(lista, 25)
    q3 = np.percentile(lista, 75)
    os = 2 * (q3 - q1) / (q3 + q1)
    return os

# Lista para almacenar los resultados de OS
os_variables78 = []
# Calcular OS para cada sublista en "variables" y guardar los resultados
for sublist in variables78:
    os_sublist = calcular_os(sublist)
    os_variables78.append(os_sublist)
# Lista para almacenar los resultados de OS
os_variables79 = []
# Calcular OS para cada sublista en "variables" y guardar los resultados
for sublist in variables79:
    os_sublist = calcular_os(sublist)
    os_variables79.append(os_sublist)

# Lista para almacenar los resultados de OS
os_novariables78 = []
# Calcular OS para cada sublista en "variables" y guardar los resultados
for sublist in novariables78:
    os_sublist = calcular_os(sublist)
    os_novariables78.append(os_sublist)
os_novariables79 = []
# Calcular OS para cada sublista en "variables" y guardar los resultados
for sublist in novariables79:
    os_sublist = calcular_os(sublist)
    os_novariables79.append(os_sublist)
os_variables=os_variables78+os_variables79    
os_novariables=os_novariables78+os_novariables79  
os_variables, os_novariables = estandarizar_lista(os_variables), estandarizar_lista(os_novariables)      
print(len(os_variables))
print(len(os_novariables))

# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))

# Graficar histogramas en el mismo gráfico
ax.hist(os_variables, bins=10, color='blue', alpha=0.5, density=True, label='s_variables')
ax.hist(os_novariables, bins=10, color='green', alpha=0.5, density=True, label='s_novariables')

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('OS')
ax.set_xlabel('Values')
ax.set_ylabel('Density')

# Ajustar el diseño
plt.tight_layout()

# Guardar el gráfico en un archivo
plt.savefig('C_OS.png')

# Mostrar la gráfica
plt.show()
plt.close()

# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(os_variables)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(os_novariables < threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine Os")


# In[ ]:


# 12. REUCLID

def calcular_distancia_euclidiana(X, Y):
    N = len(X)
    distancias = [math.sqrt((X[i + 1] - X[i])**2 + (Y[i + 1] - Y[i])**2) for i in range(N - 1)]
    distancia_media = sum(distancias) / (N - 1)
    return distancia_media

rEucliDs_variables78 = []
for i in range(len(HJD_variable78)):
    rEucliD = calcular_distancia_euclidiana(HJD_variable78[i], variables78[i])
    rEucliDs_variables78.append(rEucliD)
rEucliDs_variables79 = []
for i in range(len(HJD_variable79)):
    rEucliD = calcular_distancia_euclidiana(HJD_variable79[i], variables79[i])
    rEucliDs_variables79.append(rEucliD)
    
rEucliDs_novariables78 = []
for j in range(len(HJD_novariable78)):
    rEucliD = calcular_distancia_euclidiana(HJD_novariable78[j], novariables78[j])
    rEucliDs_novariables78.append(rEucliD)
rEucliDs_novariables79 = []
for j in range(len(HJD_novariable79)):
    rEucliD = calcular_distancia_euclidiana(HJD_novariable79[j], novariables79[j])
    rEucliDs_novariables79.append(rEucliD)
    
rEucliDs_variables=rEucliDs_variables78+rEucliDs_variables79
rEucliDs_novariables= rEucliDs_novariables78+rEucliDs_novariables79   
rEucliDs_variables, rEucliDs_novariables = estandarizar_lista(rEucliDs_variables), estandarizar_lista(rEucliDs_novariables)  
print(len(rEucliDs_variables))
print(len(rEucliDs_novariables))

# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))

# Graficar histogramas en el mismo gráfico
ax.hist(rEucliDs_variables, bins=10, color='blue', alpha=0.5, density=True, label='s_variables')
ax.hist(rEucliDs_novariables, bins=10, color='green', alpha=0.5, density=True, label='s_novariables')

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('REUCLIDS')
ax.set_xlabel('Values')
ax.set_ylabel('Density')

# Ajustar el diseño
plt.tight_layout()

# Guardar el gráfico en un archivo
plt.savefig('C_REUCLIDS.png')

# Mostrar la gráfica
plt.show()
plt.close()

# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(rEucliDs_variables)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(rEucliDs_novariables < threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine rEucliDs")


# In[ ]:


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

resultados_low_v78 = calcular_low(s_variables78, HJD78)
resultados_low_nv78 = calcular_low(s_novariables78, HJD78)
resultados_low_v79 = calcular_low(s_variables79, HJD79)
resultados_low_nv79 = calcular_low(s_novariables79, HJD79)
resultados_low_v=resultados_low_v78+resultados_low_v79
resultados_low_nv=resultados_low_nv78+resultados_low_nv79
resultados_low_v, resultados_low_nv = estandarizar_lista(resultados_low_v), estandarizar_lista(resultados_low_nv)  
print(len(resultados_low_v))
print(len(resultados_low_nv))


# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))

# Graficar histogramas en el mismo gráfico
ax.hist(resultados_low_v, bins=10, color='blue', alpha=0.5, density=True, label='s_variables')
ax.hist(resultados_low_nv, bins=10, color='green', alpha=0.5, density=True, label='s_novariables')

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('low')
ax.set_xlabel('Values')
ax.set_ylabel('Density')

# Ajustar el diseño
plt.tight_layout()
# Guardar el gráfico en un archivo
plt.savefig('C_Low.png')

# Mostrar la gráfica
plt.show()
plt.close()

# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(resultados_low_v)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(resultados_low_nv< threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine Low")


# In[ ]:


# 14. Row
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

resultados_row_v78 = calcular_row(s_variables78, HJD78)
resultados_row_nv78 = calcular_row(s_novariables78, HJD78)
resultados_row_v79 = calcular_row(s_variables79, HJD79)
resultados_row_nv79 = calcular_row(s_novariables79, HJD79)
resultados_row_v=resultados_row_v78+resultados_row_v79
resultados_row_nv=resultados_row_nv78+resultados_row_nv79
resultados_row_v, resultados_row_nv = estandarizar_lista(resultados_row_v), estandarizar_lista(resultados_row_nv)  
print(len(resultados_row_v))
print(len(resultados_row_nv))


# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))

# Graficar histogramas en el mismo gráfico
ax.hist(resultados_row_v, bins=10, color='blue', alpha=0.5, density=True, label='s_variables')
ax.hist(resultados_row_nv, bins=10, color='green', alpha=0.5, density=True, label='s_novariables')

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('row')
ax.set_xlabel('Values')
ax.set_ylabel('Density')

# Ajustar el diseño
plt.tight_layout()

# Guardar el gráfico en un archivo
plt.savefig('C_Row.png')

# Mostrar la gráfica
plt.show()
plt.close()

# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(resultados_row_v)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(resultados_row_nv< threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine row")


# In[ ]:


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


resultados_DeltaM_v78 = calcular_DeltaM(s_variables78, HJD78)
resultados_DeltaM_nv78 = calcular_DeltaM(s_novariables78, HJD78)
resultados_DeltaM_v79 = calcular_DeltaM(s_variables79, HJD79)
resultados_DeltaM_nv79 = calcular_DeltaM(s_novariables79, HJD79)
resultados_DeltaM_v=resultados_DeltaM_v78 +resultados_DeltaM_v79
resultados_DeltaM_nv=resultados_DeltaM_nv78+resultados_DeltaM_nv79
resultados_DeltaM_v, resultados_DeltaM_nv = estandarizar_lista(resultados_DeltaM_v), estandarizar_lista(resultados_DeltaM_nv)  
print(len(resultados_DeltaM_v))
print(len(resultados_DeltaM_nv))


# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))

# Graficar histogramas en el mismo gráfico
ax.hist(resultados_DeltaM_v, bins=10, color='blue', alpha=0.5, density=True, label='s_variables')
ax.hist(resultados_DeltaM_nv, bins=10, color='green', alpha=0.5, density=True, label='s_novariables')

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('DeltaM')
ax.set_xlabel('Values')
ax.set_ylabel('Density')

# Ajustar el diseño
plt.tight_layout()

# Guardar el gráfico en un archivo
plt.savefig('C_DeltaM.png')

# Mostrar la gráfica
plt.show()
plt.close()


# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(resultados_DeltaM_v)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(resultados_DeltaM_nv< threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine Delta")


# In[ ]:


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

resultados_slope_v78 = calcular_slope(s_variables78, HJD78)
resultados_slope_nv78 = calcular_slope(s_novariables78, HJD78)
resultados_slope_v79 = calcular_slope(s_variables79, HJD79)
resultados_slope_nv79 = calcular_slope(s_novariables79, HJD79)
resultados_slope_v=resultados_slope_v78+resultados_slope_v79
resultados_slope_nv=resultados_slope_nv78+resultados_slope_nv79
resultados_slope_v, resultados_slope_nv = estandarizar_lista(resultados_slope_v), estandarizar_lista(resultados_slope_nv)  
print(len(resultados_slope_v))
print(len(resultados_slope_nv))



# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))

# Graficar histogramas en el mismo gráfico
ax.hist(resultados_slope_v, bins=10, color='blue', alpha=0.5, density=True, label='s_variables')
ax.hist(resultados_slope_nv, bins=10, color='green', alpha=0.5, density=True, label='s_novariables')

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('Slope')
ax.set_xlabel('Values')
ax.set_ylabel('Density')

# Ajustar el diseño
plt.tight_layout()

# Guardar el gráfico en un archivo
plt.savefig('C_Slope.png')

# Mostrar la gráfica
plt.show()
plt.close()



# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(resultados_slope_v)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(resultados_slope_nv< threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine Slpe")


# In[ ]:


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


resultados_slope_min_v78 = calcular_slope_min(s_variables78, HJD78)
resultados_slope_min_nv78 = calcular_slope_min(s_novariables78, HJD78)
resultados_slope_min_v79 = calcular_slope_min(s_variables79, HJD79)
resultados_slope_min_nv79 = calcular_slope_min(s_novariables79, HJD79)
resultados_slope_min_v=resultados_slope_min_v78 +resultados_slope_min_v79
resultados_slope_min_nv=resultados_slope_min_nv78+resultados_slope_min_nv79
resultados_slope_min_v, resultados_slope_min_nv = estandarizar_lista(resultados_slope_min_v), estandarizar_lista(resultados_slope_min_nv)  
print(len(resultados_slope_min_v))
print(len(resultados_slope_min_nv))


# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))

# Graficar histogramas en el mismo gráfico
ax.hist(resultados_slope_min_v, bins=10, color='blue', alpha=0.5, density=True, label='s_variables')
ax.hist(resultados_slope_min_nv, bins=10, color='green', alpha=0.5, density=True, label='s_novariables')

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('Slope_min')
ax.set_xlabel('Values')
ax.set_ylabel('Density')

# Ajustar el diseño
plt.tight_layout()

# Guardar el gráfico en un archivo
plt.savefig('C_Slope_min.png')

# Mostrar la gráfica
plt.show()
plt.close()


# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(resultados_slope_min_v)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(resultados_slope_min_nv< threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine min sople")


# In[ ]:


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


resultados_r_value_v78 = calcular_r_value(s_variables78, HJD78)
resultados_r_value_nv78 = calcular_r_value(s_novariables78, HJD78)
resultados_r_value_v79 = calcular_r_value(s_variables79, HJD78)
resultados_r_value_nv79 = calcular_r_value(s_novariables79, HJD79)
resultados_r_value_v=resultados_r_value_v78+resultados_r_value_v79  
resultados_r_value_nv= resultados_r_value_nv78 +resultados_r_value_nv79                                          
resultados_r_value_v, resultados_r_value_nv = estandarizar_lista(resultados_r_value_v), estandarizar_lista(resultados_r_value_nv)  
print(len(resultados_r_value_v))
print(len(resultados_r_value_nv))


# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))

# Graficar histogramas en el mismo gráfico
ax.hist(resultados_r_value_v, bins=10, color='blue', alpha=0.5, density=True, label='s_variables')
ax.hist(resultados_r_value_nv, bins=10, color='green', alpha=0.5, density=True, label='s_novariables')

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('r_value')
ax.set_xlabel('Values')
ax.set_ylabel('Density')

# Ajustar el diseño
plt.tight_layout()

# Guardar el gráfico en un archivo
plt.savefig('C_r_value.png')

# Mostrar la gráfica
plt.show()
plt.close()


# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(resultados_r_value_v)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(resultados_r_value_nv< threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine r_value")


# In[ ]:


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


resultados_r_value_min_v78 = calcular_r_value_min(s_variables78, HJD78)
resultados_r_value_min_nv78 = calcular_r_value_min(s_novariables78, HJD78)
resultados_r_value_min_v79 = calcular_r_value_min(s_variables79, HJD79)
resultados_r_value_min_nv79 = calcular_r_value_min(s_novariables79, HJD79)
resultados_r_value_min_v=resultados_r_value_min_v78 + resultados_r_value_min_v79
resultados_r_value_min_nv=resultados_r_value_min_nv78+resultados_r_value_min_nv79
resultados_r_value_min_v , resultados_r_value_min_nv = estandarizar_lista(resultados_r_value_min_v ), estandarizar_lista(resultados_r_value_min_nv)  
print(len(resultados_r_value_min_v))
print(len(resultados_r_value_min_nv))


# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))

# Graficar histogramas en el mismo gráfico
ax.hist(resultados_r_value_min_v, bins=10, color='blue', alpha=0.5, density=True, label='s_variables')
ax.hist(resultados_r_value_min_nv, bins=10, color='green', alpha=0.5, density=True, label='s_novariables')

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('r_value_min')
ax.set_xlabel('Values')
ax.set_ylabel('Density')

# Ajustar el diseño
plt.tight_layout()

# Guardar el gráfico en un archivo
plt.savefig('C_r_value_min.png')

# Mostrar la gráfica
plt.show()
plt.close()


# In[ ]:


# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(resultados_r_value_min_v)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(resultados_r_value_min_nv< threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine min rvalue")


# In[ ]:


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


resultados_eta_v78 = calcular_eta(s_variables78, HJD78)
resultados_eta_nv78 = calcular_eta(s_novariables78, HJD78)
resultados_eta_v79 = calcular_eta(s_variables79, HJD79)
resultados_eta_nv79 = calcular_eta(s_novariables79, HJD79)
resultados_eta_v=resultados_eta_v78 +resultados_eta_v79 
resultados_eta_nv=resultados_eta_nv78+resultados_eta_nv79
resultados_eta_v ,resultados_eta_nv = estandarizar_lista(resultados_eta_v ), estandarizar_lista(resultados_eta_nv)  
print(len(resultados_eta_v))
print(len(resultados_eta_nv))



# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))

# Graficar histogramas en el mismo gráfico
ax.hist(resultados_eta_v, bins=10, color='blue', alpha=0.5, density=True, label='s_variables')
ax.hist(resultados_eta_nv, bins=10, color='green', alpha=0.5, density=True, label='s_novariables')

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('eta')
ax.set_xlabel('Values')
ax.set_ylabel('Density')

# Ajustar el diseño
plt.tight_layout()

# Guardar el gráfico en un archivo
plt.savefig('C_eta.png')

# Mostrar la gráfica
plt.show()
plt.close()


# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(resultados_eta_v)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(resultados_eta_nv< threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine ETA")


# In[ ]:


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


resultados_reDSign_v78 = calcular_reDSign(s_variables78, HJD78)
resultados_reDSign_nv78 = calcular_reDSign(s_novariables78, HJD78)
resultados_reDSign_v79 = calcular_reDSign(s_variables79, HJD79)
resultados_reDSign_nv79 = calcular_reDSign(s_novariables79, HJD79)
resultados_reDSign_v=resultados_reDSign_v78+resultados_reDSign_v79
resultados_reDSign_nv=resultados_reDSign_nv78+resultados_reDSign_nv79
resultados_reDSign_v, resultados_reDSign_nv = estandarizar_lista(resultados_reDSign_v ), estandarizar_lista(resultados_reDSign_nv) 
print(resultados_reDSign_v)
print(resultados_reDSign_nv)


# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))

# Graficar histogramas en el mismo gráfico
ax.hist(resultados_reDSign_v, bins=10, color='blue', alpha=0.5, density=True, label='s_variables')
ax.hist(resultados_reDSign_nv, bins=10, color='green', alpha=0.5, density=True, label='s_novariables')

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('reDSign')
ax.set_xlabel('Values')
ax.set_ylabel('Density')

# Ajustar el diseño
plt.tight_layout()

# Guardar el gráfico en un archivo
plt.savefig('C_reDSingn.png')

# Mostrar la gráfica
plt.show()
plt.close()


# In[ ]:


# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(resultados_reDSign_v)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(resultados_reDSign_nv< threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine reDSing")


# In[ ]:


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


resultados_rbLeon_v78 = calcular_rbLeon(s_variables78, HJD78)
resultados_rbLeon_nv78 = calcular_rbLeon(s_novariables78, HJD78)
resultados_rbLeon_v79 = calcular_rbLeon(s_variables79, HJD79)
resultados_rbLeon_nv79 = calcular_rbLeon(s_novariables79, HJD79)
resultados_rbLeon_v=resultados_rbLeon_v78+resultados_rbLeon_v79
resultados_rbLeon_nv=resultados_rbLeon_nv78 +resultados_rbLeon_nv79
resultados_rbLeon_v, resultados_rbLeon_nv = estandarizar_lista(resultados_rbLeon_v), estandarizar_lista(resultados_rbLeon_nv)
print(len(resultados_rbLeon_v))
print(len(resultados_rbLeon_nv))


# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))

# Graficar histogramas en el mismo gráfico
ax.hist(resultados_rbLeon_v, bins=10, color='blue', alpha=0.5, density=True, label='s_variables')
ax.hist(resultados_rbLeon_nv, bins=10, color='green', alpha=0.5, density=True, label='s_novariables')

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('rbLeon')
ax.set_xlabel('Values')
ax.set_ylabel('Density')

# Ajustar el diseño
plt.tight_layout()

# Guardar el gráfico en un archivo
plt.savefig('C_rbLeon.png')

# Mostrar la gráfica
plt.show()
plt.close()


# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(resultados_rbLeon_v)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(resultados_rbLeon_nv< threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine rbLeon")


# In[ ]:


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


resultados_rbLeon_sign_v78 = calcular_rbLeon_sign(s_variables78)
resultados_rbLeon_sign_nv78 = calcular_rbLeon_sign(s_novariables78)
resultados_rbLeon_sign_v79 = calcular_rbLeon_sign(s_variables79)
resultados_rbLeon_sign_nv79 = calcular_rbLeon_sign(s_novariables79)
resultados_rbLeon_sign_v=resultados_rbLeon_sign_v78+resultados_rbLeon_sign_v79
resultados_rbLeon_sign_nv=resultados_rbLeon_sign_nv78+resultados_rbLeon_sign_nv79
resultados_rbLeon_sign_v, resultados_rbLeon_sign_nv = estandarizar_lista(resultados_rbLeon_sign_v), estandarizar_lista(resultados_rbLeon_sign_nv)
print(len(resultados_rbLeon_sign_v))
print(len(resultados_rbLeon_sign_nv))


# Crear la figura y los subgráficos
fig, ax = plt.subplots(figsize=(8, 6))

# Graficar histogramas en el mismo gráfico
ax.hist(resultados_rbLeon_sign_v, bins=10, color='blue', alpha=0.5, density=True, label='s_variables')
ax.hist(resultados_rbLeon_sign_nv, bins=10, color='green', alpha=0.5, density=True, label='s_novariables')

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('rbLeon_sign')
ax.set_xlabel('Values')
ax.set_ylabel('Density')

# Ajustar el diseño
plt.tight_layout()

# Guardar el gráfico en un archivo
plt.savefig('C_rbLeon_singn.png')

# Mostrar la gráfica
plt.show()
plt.close()



# Calcular la cantidad de objetos en s_novariables antes de la superposición
threshold = np.min(resultados_rbLeon_sign_v)  # Se toma el valor mínimo de s_variables como umbral
s_novariables_before_overlap = np.sum(resultados_rbLeon_sign_nv< threshold)
print(f'Cantidad de objetos en s_novariables antes de la superposición: {s_novariables_before_overlap}')
print("Termine rbsing")


# In[ ]:


# Matrix de correlacion

# In[ ]:

print("llegue a H20")
# Inicializa el cluster de H2O y mostrar info del cluster en uso
h2o.init()
mads_s_variables, mads_s_novariables= np.array(mads_s_variables), np.array(mads_s_novariables)
resultados_variables, resultados_novariables = np.array(resultados_variables), np.array(resultados_novariables)
VART_s_variables, VART_s_novariables= np.array(VART_s_variables), np.array(VART_s_novariables)
av_s_variables, av_s_novariables= np.array(av_s_variables), np.array(av_s_novariables)
rob_s_variables, rob_s_novariables= np.array(rob_s_variables), np.array(rob_s_novariables)
DIFDER_s_variables, DIFDER_s_novariables= np.array(DIFDER_s_variables), np.array(DIFDER_s_novariables)
proy_varibles, proy_novaribles= np.array(proy_varibles), np.array(proy_novaribles)
integrals_variables, integrals_novariables = np.array(integrals_variables), np.array(integrals_novariables)
#diferencias_variables, diferencias_novariables = np.array(diferencias_variables), np.array(diferencias_novariables) 
rulds_variables, rulds_novariables= np.array(rulds_variables), np.array(rulds_novariables)
os_variables, os_novariables = np.array(os_variables), np.array(os_novariables)
rEucliDs_variables, rEucliDs_novariables = np.array(rEucliDs_variables), np.array(rEucliDs_novariables)
resultados_low_v, resultados_low_nv= np.array(resultados_low_v), np.array(resultados_low_nv)
resultados_row_v, resultados_row_nv= np.array(resultados_row_v), np.array(resultados_row_nv)
resultados_DeltaM_v, resultados_DeltaM_nv = np.array(resultados_DeltaM_v), np.array(resultados_DeltaM_nv)
resultados_slope_v, resultados_slope_nv= np.array(resultados_slope_v), np.array(resultados_slope_nv)
resultados_slope_min_v, resultados_slope_min_nv= np.array(resultados_slope_min_v), np.array(resultados_slope_min_nv)
resultados_r_value_v, resultados_r_value_nv= np.array(resultados_r_value_v), np.array(resultados_r_value_nv)
resultados_r_value_min_v , resultados_r_value_min_nv = np.array(resultados_r_value_min_v) , np.array(resultados_r_value_min_nv)
resultados_eta_v ,resultados_eta_nv= np.array(resultados_eta_v) , np.array(resultados_eta_nv) 
resultados_reDSign_v, resultados_reDSign_nv= np.array(resultados_reDSign_v), np.array(resultados_reDSign_nv)
resultados_rbLeon_v, resultados_rbLeon_nv= np.array(resultados_rbLeon_v), np.array(resultados_rbLeon_nv)
resultados_rbLeon_sign_v, resultados_rbLeon_sign_nv= np.array(resultados_rbLeon_sign_v), np.array(resultados_rbLeon_sign_nv)

# Establecer la semilla
random.seed(42)  # Puedes cambiar este número por cualquier otro valor entero
# Generar números aleatorios
aleatorio = [random.random() for _ in range(1258)]
aleatorio = np.array(aleatorio)
# Concatenar todas las variables de forma independiente
mads = np.concatenate([mads_s_variables, mads_s_novariables])
resultados = np.concatenate([resultados_variables, resultados_novariables])
VART_s = np.concatenate([VART_s_variables, VART_s_novariables])
av_s = np.concatenate([av_s_variables, av_s_novariables])
rob_s = np.concatenate([rob_s_variables, rob_s_novariables])
DIFDER_s = np.concatenate([DIFDER_s_variables, DIFDER_s_novariables])
proy = np.concatenate([proy_varibles, proy_novaribles])
integrals = np.concatenate([integrals_variables, integrals_novariables])
#diferencias = np.concatenate([diferencias_variables, diferencias_novariables])
rulds = np.concatenate([rulds_variables, rulds_novariables])
os = np.concatenate([os_variables, os_novariables])
rEucliDs = np.concatenate([rEucliDs_variables, rEucliDs_novariables])
resultados_low = np.concatenate([resultados_low_v, resultados_low_nv])
resultados_row = np.concatenate([resultados_row_v, resultados_row_nv])
resultados_DeltaM = np.concatenate([resultados_DeltaM_v, resultados_DeltaM_nv])
resultados_slope = np.concatenate([resultados_slope_v, resultados_slope_nv])
resultados_slope_min = np.concatenate([resultados_slope_min_v, resultados_slope_min_nv])
resultados_r_value = np.concatenate([resultados_r_value_v, resultados_r_value_nv])
resultados_r_value_min = np.concatenate([resultados_r_value_min_v, resultados_r_value_min_nv])
resultados_eta = np.concatenate([resultados_eta_v, resultados_eta_nv])
resultados_reDSign = np.concatenate([resultados_reDSign_v, resultados_reDSign_nv])
resultados_rbLeon = np.concatenate([resultados_rbLeon_v, resultados_rbLeon_nv])
resultados_rbLeon_sign = np.concatenate([resultados_rbLeon_sign_v, resultados_rbLeon_sign_nv])

print('Concatenar')


STYPE=[]
for i in range(len(resultados_rbLeon_sign_v)):
    STYPE.append('Variable')
for j in range(len(resultados_rbLeon_sign_nv)):
    STYPE.append('No_variable')
STYPE= np.array(STYPE)


# In[ ]:


print('Se han transcrito las variables a formato H2O')


# In[ ]:


#Creamos el H2O frame con una columna
datos = h2o.H2OFrame(python_obj=STYPE, column_names=['TYPE'], column_types=["string"])

# Variables que quieres agregar
variables = [mads, resultados, VART_s, av_s, rob_s, DIFDER_s, proy, integrals,  rulds,
             os, rEucliDs, resultados_low, resultados_row, resultados_DeltaM, resultados_slope,
             resultados_slope_min, resultados_r_value, resultados_r_value_min, resultados_eta,
             resultados_reDSign, resultados_rbLeon, resultados_rbLeon_sign, aleatorio]

# Nombres de las columnas correspondientes
nombres_columnas = ['mads', 'resultados', 'VART_s', 'av_s', 'rob_s', 'DIFDER_s', 'proy', 'integrals',
                     'rulds', 'os', 'rEucliDs', 'resultados_low', 'resultados_row',
                    'resultados_DeltaM', 'resultados_slope', 'resultados_slope_min', 'resultados_r_value',
                    'resultados_r_value_min', 'resultados_eta', 'resultados_reDSign', 'resultados_rbLeon',
                    'resultados_rbLeon_sign', 'random']

# Crear y agregar cada columna al marco de datos 'datos'
for variable, nombre_columna in zip(variables, nombres_columnas):
    print(len(variable))
    # Convertir la variable a un marco de datos H2O
    variable_h2o = h2o.H2OFrame(python_obj=variable, column_names=[nombre_columna], column_types=["float"])
    # Agregar la columna al marco de datos existente
    datos = datos.cbind(variable_h2o)

# In[ ]:


print('se ha creado el frame de H2O')
# Obtener el nombre de todas las columnas excepto la primera
columnas_numericas = datos.columns[1:]  # Excluir la primera columna 'SPECTYPE'

# Seleccionar solo las columnas numéricas
datos_numericos = datos[columnas_numericas]

# Calcular la matriz de correlación
correlation_matrix = datos_numericos.cor().as_data_frame()

correlation_matrix.to_csv('C_matriz_correlacion.csv', index=False)


# In[ ]:


print('se ha calculado la matriz de correlacion')

# Crear una figura
plt.figure(figsize=(10, 8))
# Generar un mapa de calor con la matriz de correlación
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# Ajustar la disposición para que no se corte nada
plt.tight_layout()
# Guardar la figura como un archivo PNG
plt.savefig('C_matriz_correlacion.png')
# Mostrar la figura
plt.show()
plt.close()
# Convierte la variable objetivo a factor utilizando la función asfactor()
datos['TYPE'] = datos['TYPE'].asfactor()

# Define las columnas predictoras y la variable objetivo
predictores = datos.columns[1:]  # Todas las columnas excepto la primera (SPECTYPE)
objetivo='TYPE'


# In[ ]:


train, test = datos.split_frame(ratios=[0.6], seed=42)


# In[ ]:


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



#CSV feature 
# Creamos un diccionario con los nombres de las columnas y los np.array correspondientes
data2 = {
    "Type": STYPE,
    "resultados_slope_min": resultados_slope_min,
    "proy": proy,
    "integrals": integrals,
    "resultados_r_value_min": resultados_r_value_min,
    "rob_s": rob_s,
    "resultados": resultados,
    "rEucliDs": rEucliDs,
    
}

# Creamos un DataFrame de pandas usando el diccionario
df2 = pd.DataFrame(data2)

# Guardamos el DataFrame en un archivo CSV
df2.to_csv("Features_analisisFactorial.csv", index=False)


#Cluster

# Concatenación de tus características
features_variables = np.column_stack((resultados_slope_min_v, proy_varibles, integrals_variables, resultados_r_value_min_v, rob_s_variables, resultados_variables, rEucliDs_variables))
features_novariables = np.column_stack((resultados_slope_min_nv, proy_novaribles,  integrals_novariables, resultados_r_value_min_nv, rob_s_novariables, resultados_novariables, rEucliDs_novariables))
all_features = np.vstack((features_variables, features_novariables))

# Nombres de las características
feature_names = ['resultados_slope_min','proy', 'integrals', 'resultados_r_value_min', 'rob_s', ' resultados', 'rEucliDs' ]

# Estandarización de las características
scaler = StandardScaler()
scaled_features = scaler.fit_transform(all_features)

# PCA
pca = PCA(n_components=2)  # Elige el número de componentes deseados
pca.fit(scaled_features)
pca_features = pca.transform(scaled_features)

# Clustering
kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(pca_features)

# Etiquetas para la leyenda
labels = ['Variable', 'No_Variable']

# Visualización de los clusters con leyenda
plt.figure(figsize=(8, 6))
for cluster in np.unique(clusters):
    plt.scatter(pca_features[clusters == cluster, 0], pca_features[clusters == cluster, 1], label=f'Cluster {cluster+1}: {labels[cluster]}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Clusters file b278')
plt.legend(title='Groups')
plt.show()

plt.savefig('C_Clustering.png')

# Aplicar el análisis factorial
analisis_factorial = FactorAnalysis(n_components=2)
componentes = analisis_factorial.fit_transform(scaled_features)

# Visualizar los resultados
plt.figure(figsize=(8, 6))
plt.scatter(componentes[:, 0], componentes[:, 1], cmap='viridis')
plt.title('Análisis Factorial de Características Combinadas')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Groups')
plt.grid(True)
plt.show()
plt.savefig('C_Clustering_Factorial.png')
# Imprimir las cargas factoriales
cargas_factoriales = analisis_factorial.components_
print("Cargas Factoriales:")
print(pd.DataFrame(cargas_factoriales, columns=feature_names))

print('finalice el Clustering')


#Random Forest

STYPE=[]
for i in range(len(resultados_rbLeon_sign_v)):
    STYPE.append('Variable')
for j in range(len(resultados_rbLeon_sign_nv)):
    STYPE.append('No_variable')
STYPE= np.array(STYPE)

#Creamos el H2O frame con una columna
datos1 = h2o.H2OFrame(python_obj=STYPE, column_names=['TYPE'], column_types=["string"])
print(len(datos))

# Variables que quieres agregar
variables1 = [resultados_slope_min, proy, integrals, resultados_r_value_min, rob_s, resultados, rEucliDs]
print(type(variables1))

# Nombres de las columnas correspondientes
nombres_columnas1 = ['resultados_slope_min','proy', 'integrals', 'resultados_r_value_min', 'rob_s', ' resultados', 'rEucliDs' ]
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
#train, test = datos1.split_frame(ratios=[0.8])
#train= datos1
# Lista de características y variable objetivo
features = ['resultados_slope_min','proy', 'integrals', 'resultados_r_value_min', 'rob_s', ' resultados', 'rEucliDs']
target = 'TYPE'

# Entrenar el modelo RandomForest
#rf_model = H2ORandomForestEstimator(ntrees=300, max_depth=100, nfolds=5)
#rf_model.train(x=features, y=target, training_frame=train)
#modelo_py=h2o.download_model(rf_model)
#joblib.dump(modelo_py, "RandomForest27829.pkl")
# Dividir los datos en entrenamiento y prueba
train, test = datos1.split_frame(ratios=[0.7])



# Entrenar el modelo RandomForest
rf_model = H2ORandomForestEstimator(ntrees=500, max_depth=100, nfolds=3)
rf_model.train(x=features, y=target, training_frame=train)

print('he llegado a la funcion predict_proba')


#------------------------------------------------------------------------------------

# Obtener las probabilidades de predicción
predicted_probabilities =rf_model.predict(test)

print('he aplicado predict_proba')
# Obtener las probabilidades de predicción como un DataFrame de pandas
predicted_probabilities_df = predicted_probabilities.as_data_frame()

print('lo he vuelto un df')
# Ahora puedes acceder a las probabilidades de cada clase
# El DataFrame tendrá una columna para cada clase
# La columna 'Variable' contendrá las probabilidades para la clase 'Variable'
# La columna 'No_variable' contendrá las probabilidades para la clase 'No_variable'
probabilidades_clase_variable = predicted_probabilities_df['Variable']
probabilidades_clase_no_variable = predicted_probabilities_df['No_variable']

print('he entonctrado las probabilidades por clase')

# Crear histograma para las probabilidades de galaxia
plt.figure(figsize=(8, 6))
plt.hist(probabilidades_clase_variable, bins=10, color='blue', alpha=0.7)
plt.yscale('log')  # Establecer el eje Y en escala logarítmica
plt.title('Histogram of Probabilities for Galaxies')
plt.xlabel('Probability')
plt.ylabel('Ln(Frequency)')
plt.grid(True)
plt.savefig('graph1')
plt.show()
plt.close()

# Crear histograma para las probabilidades de quasar
plt.figure(figsize=(8, 6))
plt.hist(probabilidades_clase_no_variable, bins=10, color='green', alpha=0.7)
plt.yscale('log')  # Establecer el eje Y en escala logarítmica
plt.title('Histogram of Probabilities for Quasars')
plt.xlabel('Probability')
plt.ylabel('Ln(Frequency)')
plt.grid(True)
plt.savefig('graph2')
plt.show()
plt.close()



print('termine')


h2o.cluster().shutdown()
print('Proceso finalizado')

