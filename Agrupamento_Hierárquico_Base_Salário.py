import  plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler

x=[20,  27,  21,  37,  46, 53, 55,  47,  52,  32,  39,  41,  39,  48,  48]  #idades
y=[1000,1200,2900,1850,900,950,2000,2100,3000,5900,4100,5100,7000,5000,6500]  #salarios

grafico=px.scatter(x=x, y=y)
grafico.show()

#transforando em matriz
base_salario_idade=np.array([[20,1000],[27,1200],[21,2900],[37,1850],[46,900],
                        [53,950],[55,2000],[47,2100],[52,3000],[32,5900],
                        [39,4100],[41,5100],[39,7000],[48,5000],[48,6500]])
print(base_salario_idade)
#padronizando
scaler=StandardScaler()
base_salario_idade=scaler.fit_transform(base_salario_idade)

print(base_salario_idade)

#fazendo o dendrograma

import matplotlib.pyplot as plotly
from scipy.cluster.hierarchy import dendrogram, linkage

dendrograma= dendrogram(linkage(base_salario_idade, method='ward'))
plotly.title('Dendrograma')
plotly.xlabel('Pessoas')
plotly.ylabel('Distância');
plotly.show()
#clusters ideais=3

from sklearn.cluster import AgglomerativeClustering

hc_salario=AgglomerativeClustering(n_clusters=3, linkage='ward', affinity='euclidean')
rotulos=hc_salario.fit_predict(base_salario_idade)
print(rotulos)

grafico = px.scatter(x = base_salario_idade[:,0], y = base_salario_idade[:,1], color = rotulos)
grafico.show()