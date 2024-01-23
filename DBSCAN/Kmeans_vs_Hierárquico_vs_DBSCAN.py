import  plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn import datasets

x_random, y_random= datasets.make_moons(n_samples=1500, noise=0.09)

grafico=px.scatter(x= x_random[:,0], y=x_random[:, 1])
grafico.show()


from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

kmeans=KMeans(n_clusters=2)
rotulos=kmeans.fit_predict(x_random)
grafico=px.scatter(x= x_random[:,0], y=x_random[:,1], color= rotulos)
grafico.show()
##nota que não ficou correto, kmeans nao é util nesse caso


agrupamento=AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
rotulos=agrupamento.fit_predict(x_random)
grafico=px.scatter(x= x_random[:,0], y=x_random[:,1], color= rotulos)
grafico.show()
##melhorou, entretanto o algoritmo nao classificou 100% corretamente


dbscan=DBSCAN(eps=0.115)
rotulos=dbscan.fit_predict(x_random)
grafico=px.scatter(x= x_random[:,0], y=x_random[:,1], color= rotulos)
grafico.show()
##ficou bom, apenas alguns pontos nao foram classificados
##para aplicações mais comuns como a de credito, o kmeans e o agrupamento tendem a ter um resultado melhor, mas em aplicações complexas o dbscan tem melhor desempenho em geral


