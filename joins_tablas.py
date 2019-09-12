from pyspark.sql import functions as F
from pyspark import StorageLevel


def transformacion_eliminar_duplicados_anios_balance(df_bal):
    '''
    quito los duplicados que se generan por haber más de un balance por empresas correspondiente a diferentes años
    :param df:
    :return:
    '''

    df_ejercicio_maximo = df_bal.select('id', 'clcb_ano_ejercicio') \
        .groupby(['id']).agg(F.max(F.col('clcb_ano_ejercicio')) \
                             .alias('clcb_ano_ejercicio_max')) \
        .withColumnRenamed('id', 'id_max')

    condition = [df_ejercicio_maximo.clcb_ano_ejercicio_max == df_bal.clcb_ano_ejercicio,
                 df_ejercicio_maximo.id_max == df_bal.id]

    df_balance_filtered = df_bal.join(df_ejercicio_maximo, condition, 'inner') \
        .drop('id_max', 'clcb_ano_ejercicio_max')

    df_balance_join_filtered = df_balance_filtered.dropDuplicates(['id', 'clcb_ano_ejercicio'])

    return df_balance_join_filtered


def joins_de_las_tablas_sin_relaciones(df_id, df_juri, df_bal):
    '''
    Joins de todas las tablas que tienen características
    :param spark:
    :return:
    '''
    df_jurid_ident = df_id.join(df_juri, on=['id'], how='inner')

    df_balance_filtered = transformacion_eliminar_duplicados_anios_balance(df_bal)

    df_jurid_ident = df_jurid_ident.withColumn('fechadato_year', F.col('fechadato').substr(1, 4))

    df_jurid_ident_bal = df_jurid_ident.join(df_balance_filtered, on=(['id']), how='left')

    df_jurid_ident_bal_sinduplicados = df_jurid_ident_bal.dropDuplicates(['cif', 'fechadato'])

    return df_jurid_ident_bal_sinduplicados


def joins_todas_las_tablas(df_relaciones, df_caracteristicas_ini, df_caracteristicas_fin,
                           columnas_a_seleccionar, df_matrizfilial_ini, df_matrizfilial_fin):

    df_rel_grupo = df_relaciones \
        .join(df_matrizfilial_ini,
              on=['nodo_ini'],
              how='left') \
        .persist(StorageLevel.DISK_ONLY)

    print('relaciones + matrices en nodo ini {}' \
          .format(df_rel_grupo.count()))

    df_final_ini = df_rel_grupo \
        .join(df_caracteristicas_ini,
              on=['nodo_ini', 'date_year_ini', 'date_month_ini'],
              how='left') \
        .persist(StorageLevel.DISK_ONLY)

    print('relaciones + matrices + caracteristicas en nodo ini {}' \
          .format(df_final_ini.count()))

    df_rel_grupo_fin = df_final_ini \
        .join(df_matrizfilial_fin,
              on=['nodo_fin'],
              how='left') \
        .persist(StorageLevel.DISK_ONLY)

    print('tabla junta todo ini + matrices en nodo fin {}' \
          .format(df_rel_grupo_fin.count()))

    df_final_fin = df_rel_grupo_fin \
        .join(df_caracteristicas_fin,
              on=['nodo_fin', 'date_year_ini', 'date_month_ini'],
              how='left') \
        .persist(StorageLevel.DISK_ONLY)

    print('tabla junta todo ini + matrices+ caracteristicas en nodo fin --TABLA FINAL -- {}' \
          .format(df_final_fin.count()))

    df_final_variables_seleccionadas = df_final_fin\
        .select(columnas_a_seleccionar)

    return df_final_variables_seleccionadas
