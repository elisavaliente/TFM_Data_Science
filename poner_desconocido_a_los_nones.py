from pyspark.sql import functions as F
from pyspark import StorageLevel


def cambiar_none_values_por_desconocido(df, columnas):
    '''
    Cambiar valores none a 'desconocido' en las variables indicadas
    :param df:
    :param columnas:
    :return:
    '''

    df_relaciones_none_values = None
    for i in columnas:
        df_relaciones_none_values = df \
            .withColumn(i, F.when(F.isnull(F.col(i)), 'desconocido_{}'.format(i)) \
                        .otherwise(F.col(i)))

    return df_relaciones_none_values


def rellenar_importes_con_none_values(df_nodos):

    df_media_facturacion = df_nodos.groupBy('grupo')\
        .agg(F.mean('importe').alias('media_importe_grupo'))

    df_importes_nonulos = df_nodos\
        .join(df_media_facturacion,
              on=['grupo'],
              how='left') \
        .persist(StorageLevel.DISK_ONLY)

    df_importes_nonulos = df_importes_nonulos\
        .withColumn('importe_sinnone',
                    F.when(F.isnull(F.col('importe')), F.col('media_importe_grupo')) \
                    .otherwise(F.col('importe')))

    df_media_facturacion_s = df_nodos.groupBy('sector')\
        .agg(F.mean('importe').alias('media_importe_sector'))

    df_importes_nonulos_s = df_importes_nonulos\
        .join(df_media_facturacion_s,
              on=['sector'],
              how='left') \
        .persist(StorageLevel.DISK_ONLY)

    df_importes_completos = df_importes_nonulos_s\
        .withColumn('importe_final',
                    F.when(F.isnull(F.col('importe_sinnone')), F.col('media_importe_sector')) \
                    .otherwise(F.col('importe_sinnone')))

    # para por si quedara alguno nulo todavía
    media_grupos = df_importes_completos.select(F.mean('importe_final')).toPandas()
    media_importe_todos = media_grupos['avg(importe_final)'].tolist()

    df_importes_completos_2 = df_importes_completos \
        .withColumn('importe_final_sinnulos',
                    F.when(F.isnull(F.col('importe_final')), media_importe_todos[0]) \
                    .otherwise(F.col('importe_final')))

    return df_importes_completos_2
