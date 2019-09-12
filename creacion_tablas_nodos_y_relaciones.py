from pyspark.sql import functions as F
from pyspark import StorageLevel


def creacion_tabla_nodos(df, columnas_normalized, columnas_i, columnas_f):
    '''
    :param df:
    :param columnas_normalized: columnas que se van a quedar al final
    :param columnas_i: columnas que corresponden en los nodos iniciales
    :param columnas_f: columanas que correponden en los nodos finales
    :return:
    '''

    mapping = dict(zip(columnas_i, columnas_normalized))

    nodos_ini_df = \
        df \
            .select(columnas_i) \
            .dropDuplicates() \
            .select([F.col(c).alias(mapping.get(c, c)) for c in columnas_i])

    mapping = dict(zip(columnas_f, columnas_normalized))

    nodos_fin_df = \
        df \
            .select(columnas_f) \
            .dropDuplicates() \
            .select([F.col(c).alias(mapping.get(c, c)) for c in columnas_f])

    nodos_df = \
        nodos_ini_df \
            .union(nodos_fin_df) \
            .cache()

    return nodos_df


def creacion_tabla_relaciones(df, variables_a_seleccionar):

    df_aristas = \
        df \
            .withColumnRenamed('nodo_ini', 'src') \
            .withColumnRenamed('nodo_fin', 'dst') \
            .select(*variables_a_seleccionar) \
            .persist(StorageLevel.DISK_ONLY)

    print(df_aristas)

    df_aristas_nodup = df_aristas\
        .dropDuplicates(['src', 'dst', 'anho_mes'])

    print(df_aristas_nodup.count())

    return df_aristas_nodup
