import networkx as nx
import pandas as pd
from pyspark.sql import functions as F
import poner_desconocido_a_los_nones
from pyspark import StorageLevel


def variables_centralidad(df, grupos, pd_nodos, pd_arista):
    '''
    :param centralidad:
    :param centralidad_en_su_grupo:
    :param pd_nodos:
    :return:
    '''

    centralidad = variable_centralidad_grafo_total(df)
    pd_nodos['centralidad'] = pd_nodos['nodo'].map(centralidad)

    rows = []

    centralidad_en_su_grupo = centralidad_por_grupo(grupos, pd_nodos, pd_arista)
    for k1, v1 in centralidad_en_su_grupo.items():
        for k2, v2 in v1.items():
            rows.append({'grupo': k1, 'nodo': k2, 'centralidad_ensu_grupo': v2})
    tabla_centralidad_grupo = pd.DataFrame(rows)

    pd_nodos_centralidades = pd_nodos.merge(tabla_centralidad_grupo, on=['grupo', 'nodo'], how='inner')

    return pd_nodos_centralidades


def variable_centralidad_grafo_total(df):
    '''
    centralidad para cada nodo en el grafo total
    :return:
    '''

    pd_aristas = df.select('nodo_ini', 'nodo_fin', 'score_fin').toPandas()
    pd_aristas = pd_aristas.rename(columns={'nodo_ini': 'src', 'nodo_fin': 'dst', 'score_fin': 'value'})

    grafo_total = nx.from_pandas_edgelist(pd_aristas, 'src', 'dst', ['value'])

    centralidad = nx.degree_centrality(grafo_total)

    return centralidad


def centralidad_por_grupo(grupos, pd_nodos, pd_arista):
        '''
        Función para obtener el grafo de un grupo empresarial
        :param grupos: lista de los grupos empresariales
        :param pd_nodos: tabla de los nodos
        :param pd_arista: tabla de las aristas
        :return: grafo del grupo y el grupo
        '''
        keys = []
        centra = []

        for i in grupos:
            lista = pd_nodos[pd_nodos['grupo'] == i]['nodo'].tolist()

            pd_relaciones = pd_arista[(pd_arista['src'].isin(lista)) |
                                      (pd_arista['dst'].isin(lista))]

            grafo = nx.from_pandas_edgelist(pd_relaciones, 'src', 'dst', ['value'], create_using=nx.DiGraph())
            # centralidad = nx.closeness_centrality(grafo)
            centralidad = nx.degree_centrality(grafo)

            keys.append(i)
            centra.append(centralidad)

        centralidad_en_su_grupo = dict(zip(keys, centra))

        return centralidad_en_su_grupo


def variables_modelado_nodos(df):

    pd_estadisticos = estadisticos_facturacion(df)

    pd_columnas_percentil_25_y_75 = percentil_25_y_75(pd_estadisticos, df)

    pd_proporcion_inferior_75 = calcular_proporcion_facturacion_inferior_75(pd_columnas_percentil_25_y_75)

    pd_proporcion_inferior_75_y_25 = calcular_proporcion_facturacion_inferior_25(pd_proporcion_inferior_75)

    return pd_proporcion_inferior_75_y_25


def proporcion_importe_grupo(df):

    df_importes_sinnone = poner_desconocido_a_los_nones.rellenar_importes_con_none_values(df)

    df_variablessumaimporte = df_importes_sinnone\
        .groupBy('grupo') \
        .agg(F.sum('importe_sinnone').alias('importe_total_grupo'))

    df_total_importe_grupo = df_importes_sinnone\
        .join(df_variablessumaimporte,
              on=['grupo'],
              how='left') \
        .persist(StorageLevel.DISK_ONLY)

    df_importe_sobre_grupo = df_total_importe_grupo\
        .withColumn('proporcion_importe_sobre_grupo',
                    (F.col('importe_sinnone') / F.col('importe_total_grupo')))

    return df_importe_sobre_grupo


def variable_porcentajedelgrupo_conelqueserelaciona(df):

    df_conteogrupos = df.groupBy('grupo').agg(F.countDistinct('nodo').alias('conteo_empresas'))

    df_variable_conteo_grupo = df\
        .join(df_conteogrupos,
              on=['grupo'],
              how='left') \
        .persist(StorageLevel.DISK_ONLY)

    df_porcentajedelgrupo_conelqueserelaciona = df_variable_conteo_grupo\
        .withColumn('porcentajedelgrupo_conelqueserelaciona',
                    F.col('conteo_relacion_mismogrupo') / F.col('conteo_empresas'))

    df_porcentajedelgruporelacion_nonulos = df_porcentajedelgrupo_conelqueserelaciona\
        .fillna({'porcentajedelgrupo_conelqueserelaciona': 0})

    return df_porcentajedelgruporelacion_nonulos


def calcular_proporcion_facturacion_inferior_25(pd):

    pd['menor_percentil25'] = pd.apply(lambda x: is_menor_25porciento(x['percentil25'], x['percentil25']), axis=1)

    porcentaje_menor_percentil25 = pd[['grupo', 'menor_percentil25']] \
        .groupby('grupo').agg(['sum', 'count']).reset_index()

    porcentaje_menor_percentil25['porcentaje_fact_menor_percentil_25'] = \
        (porcentaje_menor_percentil25['menor_percentil25']['sum'] / (porcentaje_menor_percentil25['menor_percentil25']['count'] - 1))

    porcentaje_menor_percentil25['porcentaje_fact_menor_percentil_25'] = \
        porcentaje_menor_percentil25['porcentaje_fact_menor_percentil_25'].fillna(0)

    porcentaje_menor_percentil25 = porcentaje_menor_percentil25[['grupo', 'porcentaje_fact_menor_percentil_25']]

    pd_variable_fact_inferior_25 = pd.\
        merge(porcentaje_menor_percentil25, how='inner', on='grupo')

    return pd_variable_fact_inferior_25


def calcular_proporcion_facturacion_inferior_75(pd):

    pd['menor_percentil75'] = pd.apply(lambda x: is_menor_25porciento(x['percentil75'], x['percentil75']), axis=1)

    porcentaje_menor_percentil75 = pd[['grupo', 'menor_percentil75']] \
        .groupby('grupo').agg(['sum', 'count']).reset_index()

    porcentaje_menor_percentil75['porcentaje_fact_menor_percentil_75'] = \
        (porcentaje_menor_percentil75['menor_percentil75']['sum'] / (
                    porcentaje_menor_percentil75['menor_percentil75']['count'] - 1))

    porcentaje_menor_percentil75['porcentaje_fact_menor_percentil_75'] = \
        porcentaje_menor_percentil75['porcentaje_fact_menor_percentil_75'].fillna(0)

    porcentaje_menor_percentil75 = porcentaje_menor_percentil75[['grupo', 'porcentaje_fact_menor_percentil_75']]

    pd_variable_fact_inferior_75 = pd. \
        merge(porcentaje_menor_percentil75, how='inner', on='grupo')

    return pd_variable_fact_inferior_75


def percentil_25_y_75(pd_estadisticos, pd_nodos):

    diccionario_importe_percentil25 = dict(zip(pd_estadisticos.grupo, pd_estadisticos['25%']))
    diccionario_importe_percentil75 = dict(zip(pd_estadisticos.grupo, pd_estadisticos['75%']))

    pd_nodos['percentil25'] = pd_nodos['grupo'].map(diccionario_importe_percentil25)
    pd_nodos['percentil75'] = pd_nodos['grupo'].map(diccionario_importe_percentil75)

    return pd_nodos


def is_menor_25porciento(importe, importepercentil25):
    if importe < importepercentil25:
        resultado = 1
    else:
        resultado = 0
    return resultado


def is_menor_75porciento(importe, importepercentil75):
    if importe < importepercentil75:
        resultado = 1
    else:
        resultado = 0
    return resultado


def estadisticos_facturacion(pd_entrada):

    pd_entrada['importe_sinnone'] = pd_entrada['importe_sinnone'].astype('float')

    list_groups = list(set(list(pd_entrada['grupo'].transpose().values)))

    pd_grupo_final = pd.DataFrame()
    pd_grupo_matriz_final = pd.DataFrame()

    for grupo in list_groups:
        pd_grupo = pd \
            .DataFrame(pd_entrada[pd_entrada['grupo'] == grupo]['importe_sinnone'].describe()).T \
            .reset_index(drop=True)

        pd_grupo['grupo'] = grupo
        pd_grupo_final = pd.concat([pd_grupo_final, pd_grupo], axis=0)

        pd_grupo_matriz = pd.DataFrame(pd_entrada[(pd_entrada['grupo'] == grupo) &
                                                  (pd_entrada['ind_filial_matriz'] == 'M')]['importe_sinnone']) \
            .reset_index(drop=True)

        pd_grupo_matriz['grupo'] = grupo
        pd_grupo_matriz_final = pd.concat([pd_grupo_matriz_final, pd_grupo_matriz], axis=0)

        pd_grupo_estadisticos = pd_grupo_final.merge(pd_grupo_matriz_final, how='inner', on='grupo')
        pd_num_empresas_grupo = pd_entrada[['grupo', 'nodo']].groupby('grupo').count().reset_index()
        pd_estadisticos = pd_num_empresas_grupo.merge(pd_grupo_estadisticos, how='inner', on='grupo')

        pd_estadisticos.rename(columns={"importe_sinnone": "importe_sinnone_matriz", "nodo": "emp"})

    return pd_estadisticos
