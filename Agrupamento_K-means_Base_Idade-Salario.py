import  plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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

#treinando
k_means_Salario=KMeans(n_clusters=3)
k_means_Salario.fit(base_salario_idade)

#achar centroides
centroides=k_means_Salario.cluster_centers_
print(centroides)

#centroides com valores reais

print(scaler.inverse_transform(k_means_Salario.cluster_centers_))

rotulos=k_means_Salario.labels_
print(rotulos)

grafico1 = px.scatter(x = base_salario_idade[:,0], y = base_salario_idade[:,1], color=rotulos)
grafico2 = px.scatter(x = centroides[:,0], y = centroides[:,1], size = [12, 12, 12])
grafico3 = go.Figure(data = grafico1.data + grafico2.data)
grafico3.show()