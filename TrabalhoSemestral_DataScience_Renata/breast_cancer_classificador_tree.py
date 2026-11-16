from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from pickle import dump

dados = pd.read_csv('breast-cancer.csv', sep=',')

dados_atributos = dados.drop(columns = ['Class'])
dados_classes = dados['Class']

dados_cat_rotulos = ['age','menopause','tumor-size','inv-nodes','node-caps','breast','breast-quad','irradiat']
dados_num = dados_atributos.drop(columns=dados_cat_rotulos)
dados_cat = dados_atributos[dados_cat_rotulos]
dados_cat_normalizados = pd.get_dummies(dados_cat, prefix=dados_cat_rotulos, dummy_na=False, dtype=int)
dados_atributos = dados_num.join(dados_cat_normalizados)
dump(dados_atributos.columns, open('breast_cancer_colunas.model','wb'))

resampler = SMOTE()
dados_atributos_balanceados, dados_classes_balanceados = resampler.fit_resample(dados_atributos, dados_classes)

normalizador = MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_atributos_balanceados)
dump(modelo_normalizador, open('breast_cancer_normalizador.model','wb'))
dados_atributos_normalizados = modelo_normalizador.fit_transform(dados_atributos_balanceados)
dados_atributos_normalizados = pd.DataFrame(dados_atributos_normalizados, columns=dados_atributos.columns)

criterion = ['gini', 'entropy','log_loss']
splitter = ['best', 'random']
min_samples_split = [int(x) for x in np.linspace(start=1, stop=10,num=5)]
decision_tree_grid={
    'criterion':criterion,
    'splitter':splitter,
    'min_samples_split': min_samples_split
}
tree = DecisionTreeClassifier()
tree_hiperparametros=RandomizedSearchCV(
    estimator=tree,
    param_distributions=decision_tree_grid,
    n_iter=10,
    cv=3,
    random_state=42,
    n_jobs=-1
)
tree_hiperparametros.fit(dados_atributos_normalizados,dados_classes_balanceados)
tree = DecisionTreeClassifier(**tree_hiperparametros.best_params_)
cancer_tree = tree.fit(dados_atributos_normalizados, dados_classes_balanceados)
dump(cancer_tree, open('breast_cancer_tree.model', 'wb'))

classes_preditas = cancer_tree.predict(dados_atributos_normalizados)
acuracia_cancer_tree = accuracy_score(dados_classes_balanceados, classes_preditas)
print('Acuracia - tree:',acuracia_cancer_tree)
