import networkx as nx

def grafos_grupos(grupos, pd_nodos, pd_arista, atributos):
    '''
    Función para obtener el grafo de un grupo empresarial
    :param grupos: lista de los grupos empresariales
    :param pd_nodos: tabla de los nodos
    :param pd_arista: tabla de las aristas
    :return: grafo del grupo y el grupo
    '''
    keys = []
    resultado = []

    for i in grupos:
        lista = pd_nodos[pd_nodos['grupo'] == i]['id'].tolist()

        pd_relaciones = pd_arista[(pd_arista['src'].isin(lista)) |
                                  (pd_arista['dst'].isin(lista))]

        grafo = nx.from_pandas_edgelist(pd_relaciones,
                                        'src', 'dst', ['value'],
                                        create_using = nx.DiGraph())
        # le damos los atributos al nodo
        nx.set_node_attributes(grafo, atributos)

        keys.append(i)
        resultado.append(grafo)

    return dict(zip(keys, resultado))


def matriz_por_grupo(grupos, pd_nodos):
    '''
    Función para la matriz de cada grupo
    :param grupos: lista de los grupos empresariales
    :param pd_nodos: tabla de los nodos
    :param pd_arista: tabla de las aristas
    :return: lista con el grupo y la matriz
    '''
    matriz_grupo = []

    for i in grupos:

        pd_empresas_grupo = pd_nodos[pd_nodos['grupo'] == i]
        matriz = pd_empresas_grupo[pd_empresas_grupo['filial_matriz'] == 'M']['id'].tolist()

        if len(matriz) == 0:
            matriz_grupo.append((i, 'no_tiene_matriz'))
        else:
            matriz_grupo.append((i, matriz[0]))

    return matriz_grupo


def grupos_conmatriz_y_sinmatriz(grupos, pd_nodos):
    matriz_cada_grupo = matriz_por_grupo(grupos, pd_nodos)
    matriz_cada_grupo_dict = {grupo: matriz for grupo, matriz in matriz_cada_grupo}

    lista_nomatriz = []
    lista_matriz = []

    for i in matriz_cada_grupo:
        if i[1] == 'no_tiene_matriz':
            lista_nomatriz.append(i)
        else:
            lista_matriz.append(i)

    print('Hay {} grupos sin matriz'.format((len(lista_nomatriz))))
    print('Hay {} grupos con matriz'.format((len(lista_matriz))))

    grupos_conmatriz = []
    grupos_sinmatriz = []

    for grupo, matriz in matriz_cada_grupo_dict.items():
        if matriz != 'no_tiene_matriz':
            grupos_conmatriz.append(grupo)
        else:
            grupos_sinmatriz.append(grupo)

    return grupos_conmatriz, grupos_sinmatriz


def matriz_central(grafo_por_grupo):
    '''
    Función para obtener el grupo empresarial con el nodo de mayor centralidad de ese grupo y si ese nodo corresponde con la matriz o no
    :param grupos_g: es la lista de grafos que pertenecen al grafo de cada grupo
    :param pd_nodos: tabla de los nodos
    :param pd_arista: tabla de las aristas
    :return: lista con el grupo empresarial, nodo con mayor centralidad y si ese nodo es la matriz o no
    '''
    resultado = []

    for key, value in grafo_por_grupo.items():

        eigen_cen = sorted(nx.degree_centrality(value).items(), key=lambda x: x[1], reverse=True)
        nif_centro = eigen_cen[0][0]

        if key == 'no_grupo':
            resultado.append((key, nif_centro, 'no_grupo'))
        elif (value.node[nif_centro]['grupo'] == key) & (value.node[nif_centro]['filial_matriz'] == 'M'):
            resultado.append((key, nif_centro, 'mayor_centr_matriz'))
        elif (value.node[nif_centro]['grupo'] == key) & (value.node[nif_centro]['filial_matriz'] == 'F'):
            resultado.append((key, nif_centro, 'mayor_centr_no_matriz(filial)'))
        else:
            resultado.append((key, nif_centro, 'mayor_centr_nogrupo'))

    return resultado


def print_group_graph(grupo, grafo, pd_nodos):

    close_centra = nx.closeness_centrality(grafo)
    colors_dict = dict()
    nodes = []
    colors = []
    max_centralidad = max(close_centra.values())
    matriz_cada_grupo = matriz_por_grupo(grupo, pd_nodos)
    matriz_cada_grupo_dict = {grupo: matriz for grupo, matriz in matriz_cada_grupo}

    for nodo, centralidad in close_centra.items():
        if (centralidad == max_centralidad) & (nodo == matriz_cada_grupo_dict[grupo]):
            color = 'yellowgreen'
        elif centralidad == max_centralidad:
            color = 'tomato'
        else:
            color = 'dodgerblue'
        nodes.append(nodo)
        colors.append(color)
        colors_dict[nodo] = color

    p = Network(height='500px', width='800px', notebook=True)
    p.add_nodes(nodes, color=colors)
    p.add_edges(grafo.edges())
    p.show_buttons(filter_ = ['physics'])
    display(p.show('{}.html'.format(grupo)))


