from fastai.vision import *
from PIL import Image
from matplotlib.pyplot import imshow
from pathlib import Path
from tqdm import tqdm
import plotly
import plotly.plotly as py
import plotly.graph_objs as go


def f1_score(y_pred:Tensor, y_true:Tensor,beta:float=1, eps:float=1e-9)->Rank0Tensor:
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2

    y_pred = y_pred.argmax(dim=1).float()
    y_true = y_true.float()
    
    TP = (y_pred*y_true).sum()
    prec = TP/(y_pred.sum()+eps)
    rec = TP/(y_true.sum()+eps)
    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
    return res.mean()


def get_coordinates_and_label(items, res, df, col='preds'):
    for i, elt in enumerate(items):
        id_im = int(elt.stem.split('-')[0])
        pred = res[i]
        #print(f'{i} working on img {id_im}, preds {pred}')
        df.loc[id_im, col] = pred
        
        
def get_geoblock(df, color, name, width=0.5):
    return go.Scattergeo(
        locationmode = 'country names',
        lon = df['coord_obs_x'],
        lat = df['coord_obs_y'],
        mode = 'markers',
        name = name,
        marker = dict( 
            size = 4, 
            symbol = 'square',
            color=color,
            line = dict(
                width=width,
                color='black'
            ),
        ))

def plot_geo_info(dfs, names, colors, iplot=True):
    data = []
    for i, df in enumerate(dfs):
        data.append(get_geoblock(df, colors[i], names[i]))
        
    layout = dict(
            title = 'Nepal data', 
            width=1000,
            height=1000,
            showlegend=True,
            geo = dict(
                scope='asia',
                showland = True,
                landcolor = "rgb(250, 250, 250)",
                subunitcolor = "rgb(217, 217, 217)",
                countrycolor = "rgb(217, 217, 217)",
                countrywidth = 0.5,
                subunitwidth = 0.5        
            ),
        )

    fig = go.Figure(data=data, layout=layout )
    if iplot:
        plotly.offline.iplot(fig, validate=False, filename='iantest')
    else:
        plotly.offline.plot(fig, validate=False, filename='iantest')       