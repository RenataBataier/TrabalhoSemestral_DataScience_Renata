import pandas as pd
from pickle import load

nova_paciente = pd.DataFrame(
    [['40-49','premeno','30-34','3-5','yes',3,'left','left_up','yes']],
    columns=['age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat']
)

dados_cat_rotulos = ['age','menopause','tumor-size','inv-nodes','node-caps','breast','breast-quad','irradiat']
nova_paciente_num = nova_paciente.drop(columns=dados_cat_rotulos)
nova_paciente_cat = nova_paciente[dados_cat_rotulos]
nova_paciente_cat_normalizada = pd.get_dummies(nova_paciente_cat, prefix=dados_cat_rotulos, dummy_na=False, dtype=int)
nova_paciente = nova_paciente_num.join(nova_paciente_cat_normalizada)
colunas = load(open('breast_cancer_colunas.model', 'rb'))
nova_paciente = nova_paciente.reindex(columns=colunas, fill_value=0)

normalizador = load(open('breast_cancer_normalizador.model', 'rb'))
nova_paciente=normalizador.transform(nova_paciente)

cancer_modelo = load(open('breast_cancer_rf.model', 'rb'))

diagnostico_nova_paciente = cancer_modelo.predict(nova_paciente)
print(diagnostico_nova_paciente)
