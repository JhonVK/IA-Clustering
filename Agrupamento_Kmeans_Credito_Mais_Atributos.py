import  plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import pandas as pd

base_credit= pd.read_csv(r'C:\Users\joaov\OneDrive\Área de Trabalho\python\CSV\credit_card_clients.csv', header=1)

print(base_credit)

#DIVIDA TOTAL
base_credit['bill_total']=base_credit['BILL_AMT1'] + base_credit['BILL_AMT2'] + base_credit['BILL_AMT3'] + base_credit['BILL_AMT4'] + base_credit['BILL_AMT5'] + base_credit['BILL_AMT6']

print(base_credit)
#limite e bill total
x_cartaoplus= base_credit.iloc[:,[1, 2, 3, 4, 5, 25]].values
print(x_cartaoplus)

#escalondando
standarti= StandardScaler()

x_cartaoplus=standarti.fit_transform(x_cartaoplus)


#achar melhor numero de clusters
wcss=[]
for i in range (1, 11):
         kmeans_cartao = KMeans(n_clusters=i, random_state=0)
         kmeans_cartao.fit(x_cartaoplus)
         wcss.append(kmeans_cartao.inertia_)

print(wcss)
grafico = px.line(x = range(1,11), y = wcss)
grafico.show()


#4 ou 5 clusters fica bom

kmeans= KMeans(n_clusters=2, random_state=0)
rotulos=kmeans.fit_predict(x_cartaoplus)##fit_predict treina e tbm faz o predict

#n da para ver grafico pois tem mais de 2 atributos
#usar pca(usada para redução de dimensionalidade)

from sklearn.decomposition import PCA

pca= PCA(n_components=2)

x_cartaoplusPCA= pca.fit_transform(x_cartaoplus)
print(x_cartaoplusPCA.shape)


#agora o grafico

grafico = px.scatter(x = x_cartaoplusPCA[:,0], y = x_cartaoplusPCA[:,1], color=rotulos)
grafico.show()

#Combinar resultados

lista_clientes = np.column_stack((base_credit, rotulos))
print(lista_clientes)

lista_clientes = lista_clientes[lista_clientes[:,26].argsort()]
print(lista_clientes)