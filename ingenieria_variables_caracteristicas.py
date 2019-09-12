from pyspark.sql import functions as F, types as T, DataFrameStatFunctions as Stat
from pyspark import StorageLevel

def variables_caracteristicas_empresas(df):

    # columna cliente o no
    column_cliente_nocliente = F.udf(variable_cliente_nocliente, T.IntegerType())

    df_relaciones_cliente_nocliente = df \
        .withColumn('cliente_nocliente_ini', column_cliente_nocliente(F.col('situ_ini'))) \
        .withColumn('cliente_nocliente_fin', column_cliente_nocliente(F.col('situ_fin')))

    # columna relacion mismo grupo o no
    df_relaciones_cliente_nocliente_tipo_relacion = \
        variable_relacion_mismo_grupo(df_relaciones_cliente_nocliente)

    # columna proporcion de clientes en el grupo
    df_relaciones_cliente_tiporel_proporcion = \
        variable_proporcion_cliente_por_grupos(df_relaciones_cliente_nocliente_tipo_relacion)

    # columna proporcion relaciones es cliente
    df_relaciones_cliente_tiporel_proporcion_proprelacioncliente = \
        variable_proporcion_cliente_en_sus_relaciones(df_relaciones_cliente_tiporel_proporcion)

    return df_relaciones_cliente_tiporel_proporcion_proprelacioncliente



def variable_cliente_nocliente(list_situacion):
    '''
    Obtener 1 y 0s en función de si es cliente de bankinter o no: 1 SI es cliente, 0 NO es cliente.
    Los None, desconocidos, entran como no clientes
    '''

    if list_situacion == 'R':
        resultado = 1
    else:
        resultado = 0

    return resultado



def variable_relacion_mismo_grupo(df):
    '''
    obtener una columna con 1 en los casos en los que la relacion sea entre empresas del grupo y 0 en lo demás
    :param df:
    :return:
    '''
    df_relacion_mismo_grupo = df.withColumn('relacion_entre_mismo_grupo',
                                            F.when((F.col('grupo_ini')) == (F.col('grupo_fin')), 1) \
                                            .otherwise(0))
    return df_relacion_mismo_grupo



def variable_proporcion_cliente_por_grupos(df):
    '''
    obtener la proporcion de empresas cliente de un grupo
    :return:
    '''

    df_proporcion = \
        df\
            .withColumn('grupo_ini',
                        F.when(F.isnull(F.col('grupo_ini')), 'no_grupo')\
                        .otherwise(F.col('grupo_ini'))) \
            .withColumn('grupo_fin',
                        F.when(F.isnull(F.col('grupo_fin')), 'no_grupo')\
                        .otherwise(F.col('grupo_fin')))


    df_proporcion_cliente_ini = df_proporcion\
        .groupBy('grupo_ini', 'date_year_ini', 'date_month_ini') \
        .agg((F.sum('cliente_nocliente_ini') / F.count('cliente_nocliente_ini')).alias('proporcion_cliente_ini'))

    df_proporcion_cliente_fin = df_proporcion\
        .groupBy('grupo_fin', 'date_year_ini', 'date_month_ini') \
        .agg((F.sum('cliente_nocliente_fin') / F.count('cliente_nocliente_fin')).alias('proporcion_cliente_fin'))

    # join ini
    df_proporcion_ini = df\
        .join(df_proporcion_cliente_ini,
                                on=['grupo_ini', 'date_year_ini', 'date_month_ini'],
                                how='left') \
        .persist(StorageLevel.DISK_ONLY)

    # join fin
    df_proporcion_cliente_por_grupo = df_proporcion_ini\
        .join(df_proporcion_cliente_fin,
              on=['grupo_fin', 'date_year_ini', 'date_month_ini'],
              how='left') \
        .persist(StorageLevel.DISK_ONLY)

    return df_proporcion_cliente_por_grupo



def variable_proporcion_cliente_en_sus_relaciones(df):
    '''
    obtener el porcentaje de cliente que hay entre las empresas con las que tiene relación
    :param df:
    :return:
    '''

    df_proporcion_relacioncliente_ini = df\
        .groupBy('nodo_ini', 'date_year_ini', 'date_month_ini') \
        .agg(((F.sum('cliente_nocliente_fin')) / (F.count('cliente_nocliente_fin'))).alias('prop_cliente_relaciones_ini'))

    df_proporcion_relacioncliente_fin = df\
        .groupBy('nodo_fin', 'date_year_ini', 'date_month_ini') \
        .agg(((F.sum('cliente_nocliente_ini')) / (F.count('cliente_nocliente_ini'))).alias('prop_cliente_relaciones_fin'))

    df_proporcion_relacioncliente = df\
        .join(df_proporcion_relacioncliente_ini,
              on=['nodo_ini', 'date_year_ini', 'date_month_ini'],
              how='left') \
        .persist(StorageLevel.DISK_ONLY)

    df_proporcion_relacioncliente_final = df_proporcion_relacioncliente\
        .join(df_proporcion_relacioncliente_fin,
              on=['nodo_fin', 'date_year_ini', 'date_month_ini'],
              how='left') \
        .persist(StorageLevel.DISK_ONLY)

    return df_proporcion_relacioncliente_final


