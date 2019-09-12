import pandas as pd
import random


def tablas_train_test(pd):

    pd_dividir = tabla_para_dividir(pd)

    grupos = pd.grupo.tolist()
    grupos_unicos = funcion_obtener_grupos_unicos_en_orden(grupos)
    print(grupos_unicos)

    random.seed(424); grupos_train = random.sample(grupos_unicos, int(0.80 * len(grupos_unicos)))
    data_train = pd_dividir[pd_dividir['grupo'].isin(grupos_train)]

    grupos_test = [x for x in grupos if x not in grupos_train]
    data_test = pd_dividir[pd_dividir['grupo'].isin(grupos_test)]

    return data_train, data_test

def funcion_obtener_grupos_unicos_en_orden(grupos):

    chequeado = []
    for x in grupos:
        if x in chequeado:
            continue
        else:
            chequeado.append(x)

    return chequeado


def tabla_para_dividir(pd):

    pd_dummies = creacion_dummies(pd)

    pd_target_b = transformacion_variable_target_en_binaria(pd_dummies)

    return pd_target_b


def creacion_dummies(dataframe):
    data_modelos = pd.get_dummies(dataframe, columns=['sector', 'tamano'], prefix=['sector', 'tamano'])
    return data_modelos


def transformacion_variable_target_en_binaria(pd):

    pd['matriz_filia_binaria'] = pd.ind_filial_matriz.apply(lambda x: binary_target(x))

    return pd


def binary_target(columna):
    if columna == 'M':
        resultado = 1
    else:
        resultado = 0
    return resultado
