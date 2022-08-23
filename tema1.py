# Librerias
from pstats import StatsProfile
import pandas as pd
import numpy as np
from scipy.stats import trim_mean
from statsmodels import robust
import matplotlib.pylab as plt
import seaborn as sns

# Rutas definidas
STATE_CSV = '/home/ruben/Documentos/Programas/practical-statistics-for-data-scientists-master/data/state.csv'
RETRASOS='/home/ruben/Documentos/Programas/practical-statistics-for-data-scientists-master/data/dfw_airline.csv'
DATOS_SP500 = '/home/ruben/Documentos/Programas/practical-statistics-for-data-scientists-master/data/sp500_data.csv'
SECTOR_SP500 = '/home/ruben/Documentos/Programas/practical-statistics-for-data-scientists-master/data/sp500_sectors.csv'

# FUNCIONES

# Funcion cargar csv y mostrar como tabla
def programa1():
    state = pd.read_csv(STATE_CSV)
    print(state.head(8))
# fin de programa1

    
# esta funcion importa los datos, y calcula la media
# la media truncada para 10% arriba y abajo
# y la mediana
def programa2():
    state = pd.read_csv(STATE_CSV)
    print("Media: ", state['Population'].mean())
    print("Media truncada: ", trim_mean(state['Population'], 0.1))
    print("Mediana: ", state['Population'].median())
# fin de programa2


#desviación estandar, rango intercuantil y desviacion absuluta mediana de la mediana
def programa3():
    state = pd.read_csv(STATE_CSV)
    desEstandar=state['Population'].std()
    rangInterQuan=state['Population'].quantile(0.75)-state['Population'].quantile(0.25)
    mad=robust.scale.mad(state['Population'])
    print('Desviación estandar: ',desEstandar)
    print('Rango intercuantil (IQR): ',rangInterQuan)
    print('Desviación absuluta mediana de la mediana: ',mad)
#Fin de programa3

##genera una tabla de frecuencia que divide el rango de una variable
#en segmentos igualmente espaciados y nos dice que valores caen dentro
def programa4():
    state = pd.read_csv(STATE_CSV)
    binnedPopulation=pd.cut(state['Population'], 10)
    tabla=binnedPopulation.value_counts()
    print(tabla)
#Fin de programa4

#Genera un histograma
def programa5():
    state = pd.read_csv(STATE_CSV)
    ax=(state['Population']/1000000).plot.hist(figsize=(4,4))
    ax.set_xlabel('Población (millones)')
    ax.set_ylabel('Frecuencia')
    #Para dibujar en ventana
    plt.tight_layout()
    plt.show()
#Fin de programa5

#Muestra algunos quantiles (percentiles), porcentaje de la muestra que toma ese valor o inferior
def programa6():
    state = pd.read_csv(STATE_CSV)
    percentil=state['Murder.Rate'].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print(percentil)
#Fin de programa6

#Diagrama de caja
def programa7():
    state = pd.read_csv(STATE_CSV)
    ax=(state['Population']/1000000).plot.box()
    ax.set_ylabel('Poblacion (millones)')
    plt.tight_layout()
    plt.show()
#Fin programa7

#histograma
def programa8():
    state=pd.read_csv(STATE_CSV)
    ax=(state['Population']/1000000).plot.hist(figsize=(4,4))
    ax.set_xlabel('Población (millones)')
    plt.tight_layout()
    plt.show()
#Fin de programa 8


#Diagrama y estimación de la curva de densidad
def programa9():
    state=pd.read_csv(STATE_CSV)
    ax=state['Murder.Rate'].plot.hist(density=True, xlim=[0,12], bins=range(1,12))
    state['Murder.Rate'].plot.density(ax=ax)
    ax.set_xlabel('asesinatos por cada 1000000')
    plt.tight_layout()
    plt.show()
#Fin de programa 9


#Gráfico de barras
def programa10():
    dfw=pd.read_csv(RETRASOS)
    print(100 * dfw / dfw.values.sum()) ##Imprimo en consola el porcentaje de retrasos por causa
    ax=dfw.transpose().plot.bar(figsize=(4,4), legend=False)
    ax.set_xlabel('Causa del retraso')
    ax.set_ylabel('Casos')
    plt.tight_layout()
    plt.show()
#Fin de programa 10

#Presento matrices de correlación como diagramas de calor
def programa11():
    
    ##Cargo los datos
    sp500_sym = pd.read_csv(SECTOR_SP500, header=0)
    print(sp500_sym.shape)
    sp500_px = pd.read_csv(DATOS_SP500, header=0, index_col=0) #primera fila es el nombre de las columnas
    print(sp500_px.shape)

    #filtro los datos de simbolos de las empresas de telecomunicaciones
    telecomSymbols = sp500_sym[sp500_sym['sector'] == 'telecommunications_services']['symbol']
    
    #filtro los datos para coger solo los simbolos de empresas a partir de la fecha pedida
    telecom = sp500_px.loc[sp500_px.index >= '2012-07-01', telecomSymbols]
    
    telecom.corr()
    print(telecom)
    
    etfs = sp500_px.loc[sp500_px.index > '2012-07-01', sp500_sym[sp500_sym['sector'] == 'etf']['symbol']]
    print(etfs.head())

    fig, ax = plt.subplots(figsize=(5, 4))
    ax = sns.heatmap(etfs.corr(), vmin=-1, vmax=1, cmap=sns.diverging_palette(20, 220, as_cmap=True),ax=ax)

    plt.tight_layout()
    plt.show()
#Fin de programa 11


def pruebas():
    COLUMNS = ['age','workclass', 'fnlwgt', 'education', 'education_num', 'marital',
           'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
           'hours_week', 'native_country', 'label']
    #Cargo el csv asignando nombre a cada columna
    prueba=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', names=COLUMNS) 
    
    print(prueba.shape) #imprime el número de filas y columnas respectivametne
    
    print(prueba['marital'])
# FIN DE FUNCIONES

# Funcion main


def main():
    programa11()


if __name__ == '__main__':
    main()
