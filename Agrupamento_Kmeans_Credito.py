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
x_cartao= base_credit.iloc[:,[1, 25]].values
print(x_cartao)

#padronizando
scaler=StandardScaler()
x_cartao=scaler.fit_transform(x_cartao)

#achar melhor numero de clusters
wcss=[]
for i in range (1, 11):
         kmeans_cartao = KMeans(n_clusters=i, random_state=0)
         kmeans_cartao.fit(x_cartao)
         wcss.append(kmeans_cartao.inertia_)

print(wcss)
grafico = px.line(x = range(1,11), y = wcss)
grafico.show()

#4 ou 5 clusters fica bom

kmeans= KMeans(n_clusters=4, random_state=0)
rotulos=kmeans.fit_predict(x_cartao)               ##fit_predict treina e tbm faz o predict

grafico = px.scatter(x = x_cartao[:,0], y = x_cartao[:,1], color=rotulos)
grafico.show()

lista_clientes = np.column_stack((base_credit, rotulos))
print(lista_clientes)