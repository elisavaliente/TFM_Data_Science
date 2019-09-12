import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import mannwhitneyu



def test_hipotesis(variable_matrices, variable_filiales):

    data1 = np.array(variable_matrices).astype('double')
    data2 = np.array(variable_filiales).astype('double')

    # compare samples
    stat, p = mannwhitneyu(data1, data2, alternative='two-sided')
    print('Statistics=%.3f, p=%.15f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')

    data = np.concatenate((data1, data2), axis=None)
    labels = np.concatenate((np.ones(len(data1)), np.zeros(len(data2))), axis=None)

    # Calculamos la diferencia de medias original
    median1 = np.median(data[labels == 1])
    median2 = np.median(data[labels == 0])
    diff = median1 - median2

    # Hacemos shuffling
    tmp_diff_acumulado = []
    num_iterations = 10000
    num_diff_greater = 0
    for i in range(num_iterations):
        np.random.shuffle(labels)
        tmp_median1 = np.median(data[labels == 1])
        tmp_median2 = np.median(data[labels == 0])
        tmp_diff = tmp_median1 - tmp_median2
        tmp_diff_acumulado.append(tmp_diff)
        if diff <= tmp_diff:
            num_diff_greater += 1

    # Grafico con porcentajes antiguos
    sns.set(style="whitegrid")
    # Las instrucciones a continuacion terminan con ';' para evitar que en jupyter salgan salidas intermedias (e.g., texto y datos)
    # Ver https://stackoverflow.com/a/51629429
    fig, ax = plt.subplots()
    fig.set_size_inches([20, 10])
    sns.distplot(tmp_diff_acumulado, color='green');
    plt.axvline(diff)

    ax.set_title('Distribución del conteo de relaciones con empresas del grupo', fontsize=15)
    ax.set(xlabel='Distribution ', ylabel='Frequency');

    return (num_diff_greater / num_iterations)


def dibujar_ditribuciones(variable_matriz, variable_filial):
    # Grafico con porcentajes antiguos
    sns.set(style="whitegrid")

    # Las instrucciones a continuacion terminan con ';' para evitar que en jupyter salgan salidas intermedias (e.g., texto y datos)
    # Ver https://stackoverflow.com/a/51629429
    fig, ax = plt.subplots()
    fig.set_size_inches([20, 10])
    sns.kdeplot(variable_matriz, color='blue');
    sns.kdeplot(variable_filial, color='red');
    # sns.kdeplot(pd_newvariables_4['importe_total_grupo'], color = 'green');

    ax.set_title('Distribución de la facturación de las empresas', fontsize=15)
    ax.set(xlabel='Distribution ', ylabel='Frequency');