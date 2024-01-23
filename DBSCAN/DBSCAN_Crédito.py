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

#padronizando
scaler=StandardScaler()
x_cartao=scaler.fit_transform(x_cartao)
print(x_cartao)

from sklearn.cluster import DBSCAN

dbscan=DBSCAN(eps=0.38, min_samples=5)

dbscan.fit(x_cartao)

rotulos=dbscan.labels_

print(np.unique(rotulos, return_counts=True))


grafico = px.scatter(x = x_cartao[:,0], y = x_cartao[:,1], color = rotulos)
grafico.show()

