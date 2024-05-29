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


# In[3]:


archivo1 = fits.open('Medias-Desviacion278.fits')
archivo = fits.open('TCampob278.fits')
s_variables78=np.array(archivo1[9].data)
HJD78=archivo[1].data
archivo2 = fits.open('Medias-Desviacion279.fits')
s_variables79=np.array(archivo2[10].data)
archivo3 = fits.open('TCampo_b279.fits')
HJD79=archivo3[1].data
archivo1.info()
archivo2.info()


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
resultados_slope_min_v79 = calcular_slope_min(s_variables79, HJD79)
resultados_slope_min_v=resultados_slope_min_v78 +resultados_slope_min_v79
resultados_slope_min_v= estandarizar_lista(resultados_slope_min_v)
print(len(resultados_slope_min_v))
print(type(resultados_slope_min_v))

# In[ ]:


#7. PROY 2

# Función para eliminar los valores nan de una lista
def remove_nan(lst):
    return [x for x in lst if not np.isnan(x)]

# Eliminar los valores nan de las listas en s_variables y guardar en HDJ_variables
variables78 = [remove_nan(lst) for lst in s_variables78]
variables79 = [remove_nan(lst) for lst in s_variables79]


# Función para eliminar np.nan de HJD por cada sublista de s_variables
def remove_nan_in_HJD(HJD, s_variables):
    result = []
    for sublist in s_variables:
        valid_indices = [i for i, val in enumerate(sublist) if not np.isnan(val)]
        result.append([HJD[i] for i in valid_indices])
    return result

# Eliminar np.nan en HJD por cada sublista de s_variables
HJD_variable78 = remove_nan_in_HJD(HJD78, s_variables78)
HJD_variable79 = remove_nan_in_HJD(HJD79, s_variables79)


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
    
    

    
proy_varibles= proy_varibles78+ proy_varibles79
proy_varibles= estandarizar_lista(proy_varibles)

print(len(proy_varibles))
print(type(proy_varibles))

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
    

integrals_variables=integrals_variables78+integrals_variables79
integrals_variables= estandarizar_lista(integrals_variables)
print(len(integrals_variables))
print(type(integrals_variables))

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
resultados_r_value_min_v79 = calcular_r_value_min(s_variables79, HJD79)
resultados_r_value_min_v=resultados_r_value_min_v78 + resultados_r_value_min_v79
resultados_r_value_min_v = estandarizar_lista(resultados_r_value_min_v )
print(len(resultados_r_value_min_v))
print(type(resultados_r_value_min_v))

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

def calcular_robAbbe_separado(s_variables, c=1.345):
    resultados_s_variables = []


    for sublist_vars in s_variables:
        robAbbe_vars = calcular_robAbbe(sublist_vars, c=c)
        resultados_s_variables.append(np.nanmean(robAbbe_vars))  # Agregar la media de los resultados


    return resultados_s_variables

rob_s_variables78= calcular_robAbbe_separado(s_variables78)
rob_s_variables79 = calcular_robAbbe_separado(s_variables79)
rob_s_variables=rob_s_variables78+rob_s_variables79
print(len(rob_s_variables))
print(type(rob_s_variables))

# In[ ]:


# 2. Mediana

def calcular_mediana_por_objeto(s_variables):
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
   

   return resultados_variables

resultados_variables78= calcular_mediana_por_objeto(s_variables78)
resultados_variables79 = calcular_mediana_por_objeto(s_variables79)
resultados_variables=resultados_variables78+resultados_variables79
resultados_variables = estandarizar_lista(resultados_variables)
print(len(resultados_variables))
print(type(resultados_variables))

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
    

    
rEucliDs_variables=rEucliDs_variables78+rEucliDs_variables79 
rEucliDs_variables = estandarizar_lista(rEucliDs_variables)
print(len(rEucliDs_variables))
print(type(rEucliDs_variables))

# In[ ]:


# Concatenar todas las variables de forma independiente
resultados = np.array(resultados_variables)
print(type(resultados))
rob_s = np.array(rob_s_variables)
print(type(rob_s))
proy = np.array(proy_varibles)
print(type(proy))
integrals = np.array(rEucliDs_variables)
print(type(integrals))
resultados_slope_min = np.array(resultados_slope_min_v)
print(type(resultados_slope_min))
resultados_r_value_min = np.array(resultados_r_value_min_v)
print(type(resultados_r_value_min))
rEucliDs=np.array(rEucliDs_variables)
print(type(rEucliDs))

#-----------------------------------------------
print('inicie h2o')
h2o.init()

#Creamos el H2O frame con una columna
datos = h2o.H2OFrame(python_obj=resultados , column_names=['resultados'], column_types=["float"])

# Variables que quieres agregar
variables = [rob_s, proy, integrals, resultados_slope_min, resultados_r_value_min, rEucliDs]

# Nombres de las columnas correspondientes
nombres_columnas = ['rob_s', 'proy', 'integrals', 'resultados_slope_min', 'resultados_r_value_min', 'rEucliDs']

# Crear y agregar cada columna al marco de datos 'datos'
for variable, nombre_columna in zip(variables, nombres_columnas):
    print(len(variable))
    # Convertir la variable a un marco de datos H2O
    variable_h2o = h2o.H2OFrame(python_obj=variable, column_names=[nombre_columna], column_types=["float"])
    # Agregar la columna al marco de datos existente
    datos = datos.cbind(variable_h2o)

print('se ha creado el frame de H2O')

#------------------------------------------------

# Definir los datos de características (features) y etiquetas
#matrix = np.array([resultados_slope_min, proy, integrals, resultados_r_value_min, rob_s, resultados, rEucliDs])
#matrix = matrix.T
#print(type(matrix))

# Cargar el modelo desde el archivo .pkl
#model = joblib.load('RandomForest27829.pkl')
model = joblib.load('RandomFinal27829.pkl')
print('llegue a prediccion')
# Realizar predicciones
#predictions = model.predict(matrix)
predictions = model.predict(datos)

print('predije')
# Crear un diccionario para almacenar los recuentos de cada valor único
recuentos = {}

# Contar la cantidad de elementos de cada valor único
for valor in predictions:
    if valor in recuentos:
        recuentos[valor] += 1
    else:
        recuentos[valor] = 1

# Imprimir los recuentos
for valor, cantidad in recuentos.items():
    print("Valor:", valor, "Cantidad:", cantidad)


print('termine')



