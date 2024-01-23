import  plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler

x=[20,  27,  21,  37,  46, 53, 55,  47,  52,  32,  39,  41,  39,  48,  48]  #idades
y=[1000,1200,2900,1850,900,950,2000,2100,3000,5900,4100,5100,7000,5000,6500]  #salarios


#transforando em matriz
base_salario_idade=np.array([[20,1000],[27,1200],[21,2900],[37,1850],[46,900],
                        [53,950],[55,2000],[47,2100],[52,3000],[32,5900],
                        [39,4100],[41,5100],[39,7000],[48,5000],[48,6500]])
print(base_salario_idade)
#padronizando
scaler=StandardScaler()
base_salario_idade=scaler.fit_transform(base_salario_idade)

print(base_salario_idade)

from sklearn.cluster import DBSCAN

dbscan_salario= DBSCAN(eps=0.95, min_samples=2)
dbscan_salario.fit(base_salario_idade)
#mostra a qual cluster o registro pertence
rotulos= dbscan_salario.labels_
print(rotulos)


grafico = px.scatter(x = base_salario_idade[:,0], y = base_salario_idade[:,1], color = rotulos)
grafico.show()