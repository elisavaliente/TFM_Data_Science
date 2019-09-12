from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import pickle
import shap
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt



def modelos(train, test):
    train = train.copy(deep=True)
    test = test.copy(deep=True)

    target_train = train.loc[:, 'matriz_filia_binaria']
    data_train_sintarget = train.drop('matriz_filia_binaria', axis=1)

    dict_modelos = {
        'Decision tree': build_modelo_arbol_decision(),
        'Regresion logística': build_modelo_reg_logistica(),
        'SVM': build_modelo_svm(),
        'RandomForest': build_modelo_random_forest()
    }

    list_modelos_fitted = [(key, model.fit(data_train_sintarget, target_train.ravel()))
                           for key, model in dict_modelos.items()]

    for i in list_modelos_fitted:
    # save the model to disk
        filename = './modelo_pruebas/finalized_model{}.sav'.format(i[0])
        pickle.dump(i[1], open(filename, 'wb'))

    return evaluacion_modelos(test, list_modelos_fitted)


def build_modelo_reg_logistica():
    return LogisticRegression()


def build_modelo_arbol_decision():
    return DecisionTreeClassifier()


def build_modelo_svm():
    kernel_types = ['linear', 'rbf']
    # podemos probar tambien con otros kernel como: 'poly', 'sigmoid'

    C_range = [1, 10]

    degree_range = [1]

    parametros = {'kernel': kernel_types,
                  'C': C_range,
                  'degree': degree_range,
                  }

    grid = GridSearchCV(estimator=svm.SVC(probability=True),
                        param_grid=parametros,
                        cv=5,
                        scoring='accuracy',
                        refit=True)

    return make_pipeline(preprocessing.StandardScaler(), grid)


def build_modelo_random_forest():
    return RandomForestClassifier(n_estimators=500, max_depth=3, random_state=123)


def evaluacion_modelos(test, modelos):
    plt.figure(figsize=(14, 14))
    plt.plot([0, 1], [0, 1], 'r--');

    target_test = test.loc[:, 'matriz_filia_binaria'].ravel()
    data_test_sintarget = test.drop('matriz_filia_binaria', axis=1)

    for model_name, model in modelos:
        pred_target = model.predict(data_test_sintarget)
        pred_proba_target = model.predict_proba(data_test_sintarget)[:, 1]

        confusion_matrix_arbol = confusion_matrix(target_test, pred_target)
        print('La matriz de confusión del modelo {} es: {}'.format(model_name, confusion_matrix_arbol))
        print(classification_report(target_test, pred_target))

        plot_roc_auc(te_real_target=target_test,
                     te_pred_target=pred_target,
                     te_proba_target=pred_proba_target,
                     model_label="{}, auc=".format(model_name));

    plt.legend(loc=0);


def plot_roc_auc(te_real_target, te_pred_target, te_proba_target, model_label):
    logit_roc_auc = roc_auc_score(te_real_target, te_proba_target)
    fpr1, tpr1, thresholds1 = roc_curve(te_real_target, te_proba_target)
    plt.plot(fpr1, tpr1, label=model_label + str(round(logit_roc_auc, 2)));


def feature_importance(data_test_sintarget, columnas, modelo):
    pd_shap = to_cat(data_test_sintarget, columnas)

    # load JS visualization code to notebook
    shap.initjs()
    explainer = shap.TreeExplainer(modelo)

    # para ver la feature importance de predecir el 1
    shap_values = explainer.shap_values(pd_shap)[1]

    grafico_shap = shap.summary_plot(shap_values, pd_shap, max_display=50)

    return grafico_shap


def to_cat(pd_df, list_cat_cols):
    for c in list_cat_cols:
        pd_df[c] = pd_df[c].astype('category')
    return pd_df


def plot_learning_curve(estimator, title, X, y, ylim=[0.67, 0.8],
                        cv=ShuffleSplit(n_splits=5, train_size=0.5, test_size=0.5, random_state=1993),
                        n_jobs=1, train_sizes=[0.1, 0.3, 0.5, 0.7, 0.9, 1.]):
    """
    Generar learning_curve
    Input:
        estimator: parámetros del modelo
        title: título del gráfico
        X: variables explicativas del modelo
        y: variable objetivo
        ylim:
        cv: semilla
        n_jobs:
        train_sizes: trozos en los que se quiere dividir la muestra (en porcentajes)

    Output:
        gráfico con las 2 curvas de entrenamiento
    """
    plt.figure(figsize=(15, 10))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='roc_auc')
    print(X.shape)
    print(y.shape)
    print('train_scores', train_scores[:10])
    print('test_scores', test_scores[:10])
    print('train_sizes', train_sizes[:10])
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def roc_cero_uno(y_true=None, y_predicted_proba=None, size_figure=[15, 5]):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función roc_cero_uno: recibe los valores reales y las predicciones de probabilidad de la target
    y representa el mejor punto de corte o umbral para asignar 0s o 1s según las predicciones de
    probabilidad
    ----------------------------------------------------------------------------------------------------------
        - Inputs:
            -- y_true: valores verdaderos de la target
            -- y_predicted_proba: valores de las predicciones de probabilidad
            -- size_figure: lista con el tamaño de la representación
        - Output: Representación de la distancia euclídea entre el punto perfecto [0,1] y el obtenido con
        el modelo. Además, se obtiene el mejor umbral y el mejor punto de la curva ROC.
        - Return:
            -- 0: sin errores
            -- 1: con errores
    '''
    #     logging.info(u'Función roc_cero_uno inicio')
    #     start = time.time()
    if ((y_true is None) or (y_predicted_proba is None)):
        print(u'\nFalta pasar argumentos a la función')
        return 1
    fpr, tpr, thresholds = roc_curve(y_true, y_predicted_proba[:, 1])

    best = [0, 1]
    dist = []
    for (x, y) in zip(fpr, tpr):
        dist.append([euclidean([x, y], best)])

    bestPoint = [fpr[dist.index(min(dist))], tpr[dist.index(min(dist))]]

    bestCutOff1 = thresholds[list(fpr).index(bestPoint[0])]
    bestCutOff2 = thresholds[list(tpr).index(bestPoint[1])]
    print('\n**********************************************************************')
    print(
        '\nMejor punto en la curva ROC: TPR = {:0.3f}%, FPR = {:0.3f}%'.format(bestPoint[1] * 100, bestPoint[0] * 100))
    print('\nMejor umbral: {:0.4f}'.format(bestCutOff1))
    print('\n**********************************************************************')

    plt.plot(dist)
    plt.xlabel('Index')
    plt.ylabel('Euclidean Distance to the perfect [0,1]')
    fig = plt.gcf()
    fig.set_size_inches(size_figure)
    return bestCutOff1


def probabilidad_a_pred_umbral(modelo_entrenado=None, X_test=None, thr=0.5):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función probabilidad_a_pred_umbral:
    ----------------------------------------------------------------------------------------------------------
        - Inputs:
            -- modelo_entrenado: modelo con el que predecir el test y obtener probabilidades
            -- X_test: variables independientes
            -- thr: umbral con el que binarizar las probabilidades
        - Return:
            -- final: lista que contiene las predicciones binarizadas usando el umbral
            -- 1: flag indicador de errores
    '''
    if ((modelo_entrenado is None) or (X_test is None)):
        print(u'\nFaltan por pasar argumentos a la función')
        return 1
    prob = modelo_entrenado.predict_proba(X_test)[:, 1]
    final = []
    for p in prob:
        if p >= thr:
            final.append(1)
        else:
            final.append(0)
    return final