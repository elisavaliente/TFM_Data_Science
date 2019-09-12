from pyspark.sql import functions as F
from pyspark import StorageLevel

def variables_modelado_relaciones(df):

    df_variables_nuevas = variable_proporcion_relaciones_empresasgrupo(df)

    df_variables_nuevas = proporcion_del_grupo_conlasque_se_relaciona(df_variables_nuevas)

    return df_variables_nuevas



def variable_proporcion_relaciones_empresasgrupo(df):
    '''
    Obtener la proporcion de relaciones que son con empresas del grupo
    :param df:
    :return:
    '''

    df_proporcion_relacionesgrupo_ini = df\
        .groupBy('nodo_ini') \
        .agg((F.sum('relacion_entre_mismo_grupo') / F.count('relacion_entre_mismo_grupo')) \
             .alias('proporcion_relaciones_congrupo_ini'))

    df_proporcion_relacionesgrupo_fin = df\
        .groupBy('nodo_fin') \
        .agg((F.sum('relacion_entre_mismo_grupo') / F.count('relacion_entre_mismo_grupo')) \
             .alias('proporcion_relaciones_congrupo_fin'))

    df_proporcion_relacionesgrupo = df\
        .join(df_proporcion_relacionesgrupo_ini,
                                            on=['nodo_ini'],
                                            how='left') \
        .persist(StorageLevel.DISK_ONLY)

    df_proporcion_relacionesgrupo = df_proporcion_relacionesgrupo\
        .join(df_proporcion_relacionesgrupo_fin,
              on=['nodo_fin'],
              how='left') \
        .persist(StorageLevel.DISK_ONLY)

    return df_proporcion_relacionesgrupo



def proporcion_del_grupo_conlasque_se_relaciona(df):
    '''
    Proporción de las empresas de su grupo con las que se relaciona, es decir, solo de las del grupo con
    cuántas tiene relación no en proporción al total
    :param df:
    :return:
    '''

    relaciones_empresas_grupo_ini = df \
        .filter(F.col('grupo_ini') == F.col('grupo_fin')) \
        .groupBy('nodo_ini').agg(F.countDistinct('nodo_fin')\
                                 .alias('conteo_relacion_mismogrupo_ini'))

    df_proporcion_del_grupo_ini = df\
        .join(relaciones_empresas_grupo_ini,
              on = ['nodo_ini'],
              how = 'left') \
        .persist(StorageLevel.DISK_ONLY)


    relaciones_empresas_grupo_fin = df \
        .filter(F.col('grupo_ini') == F.col('grupo_fin')) \
        .groupBy('nodo_fin').agg(F.countDistinct('nodo_ini')\
                                 .alias('conteo_relacion_mismogrupo_fin'))

    df_proporcion_del_grupo_fin = df_proporcion_del_grupo_ini\
        .join(relaciones_empresas_grupo_fin,
              on = ['nodo_fin'],
              how = 'left') \
        .dropDuplicates(['nodo_ini', 'nodo_fin', 'date_year_ini', 'date_month_ini']) \
        .persist(StorageLevel.DISK_ONLY)

    df_proporcion_del_grupo = df_proporcion_del_grupo_fin\
        .fillna({'conteo_relacion_mismogrupo_ini': 0,
                 'conteo_relacion_mismogrupo_fin': 0})

    return df_proporcion_del_grupo




