import numpy as np 
import pandas as pd
import plotly.express as px 
from sklearn import cluster 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler 
import plotly.graph_objects as go 

data = pd.read_csv('CREDIT_CLUSTERING\\Creadit_clustering.csv') 
data = data.dropna()
clustering_data = data[['BALANCE','PURCHASES', 'CREDIT_LIMIT']]
for i in clustering_data.columns: 
    MinMaxScaler(i)

#build the model
kmeans = KMeans(n_clusters = 5)
cluster = kmeans.fit_predict(clustering_data)
data['CREDIT_CARD_SEGMENTS'] = cluster 
#transform the cluster into string 
data['CREDIT_CARD_SEGMENTS'] = data['CREDIT_CARD_SEGMENTS'].map({0 : "cluster 1",
                                                               1 : "cluster 2",
                                                               2 : "cluster 3",
                                                               3 : "cluster 4",
                                                               4 : "cluster 5"})
fig = go.Figure()
for i in list(data['CREDIT_CARD_SEGMENTS'].unique()):
    fig.add_trace(go.Scatter3d(x=data[data['CREDIT_CARD_SEGMENTS']==i]['BALANCE'],
                               y=data[data['CREDIT_CARD_SEGMENTS']==i]['PURCHASES'],
                               z=data[data['CREDIT_CARD_SEGMENTS']==i]['CREDIT_LIMIT'],
                               mode = 'markers', marker_size = 6, marker_line_width = 1,
                               name = str(i)))
fig.update_traces(hovertemplate = 'BALANCE: %{x} <br>PURCHASES %{y} <br>CREDIT_LIMIT: %{z}')
fig.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                   scene = dict(xaxis=dict(title = 'BALANCE', titlefont_color = 'black'),
                                yaxis=dict(title = 'PURCHASES', titlefont_color = 'black'),
                                zaxis=dict(title = 'CREDIT_LIMIT', titlefont_color = 'black')),
                   font = dict(family = "Gilroy", color  = 'black', size = 12))
fig.show()
PLOT = go.Figure()
for i in list(data["CREDIT_CARD_SEGMENTS"].unique()):
    

    PLOT.add_trace(go.Scatter3d(x = data[data["CREDIT_CARD_SEGMENTS"]== i]['BALANCE'],
                                y = data[data["CREDIT_CARD_SEGMENTS"] == i]['PURCHASES'],
                                z = data[data["CREDIT_CARD_SEGMENTS"] == i]['CREDIT_LIMIT'],                        
                                mode = 'markers',marker_size = 6, marker_line_width = 1,
                                name = str(i)))
PLOT.update_traces(hovertemplate='BALANCE: %{x} <br>PURCHASES %{y} <br>DCREDIT_LIMIT: %{z}')

    
PLOT.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                   scene = dict(xaxis=dict(title = 'BALANCE', titlefont_color = 'black'),
                                yaxis=dict(title = 'PURCHASES', titlefont_color = 'black'),
                                zaxis=dict(title = 'CREDIT_LIMIT', titlefont_color = 'black')),
                   font = dict(family = "Gilroy", color  = 'black', size = 12))    
PLOT.show()

