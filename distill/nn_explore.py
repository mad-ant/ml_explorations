import streamlit as st
import altair as alt
import pandas as pd
import torch
import numpy as np
import torch.nn as nn


def plot_scatter(ax, X, title = ''):
    ax.set_box_aspect(1)
    ax.scatter(X[:,0],X[:,1],c=x[:,0],cmap=plt.cm.Spectral,s=20)
    ax.axis('off')
    ax.set_title(title)
def plot_bases(ax, bases, width = 0.04):
    bases = bases.cpu()
    bases[2:] -= bases[:2]
    ax.arrow(*bases[0], *bases[2], width=width, color=(1,0,0), alpha=1., length_includes_head=True)
    ax.arrow(*bases[1], *bases[3], width=width, color=(0,1,0), alpha=1., length_includes_head=True)




nl_txt = st.sidebar.selectbox('NonLinearity',['ELU','Hardshrink','Hardsigmoid','Hardtanh','Hardswish','LeakyReLU','LogSigmoid','PReLU','ReLU','ReLU6','RReLU','SELU','CELU','GELU','Sigmoid','SiLU','Softplus','Softshrink','Softsign','Tanh','Tanhshrink'],index=8) # 'MultiheadAttention' , 'Threshold'
visualize = st.sidebar.button("Visualize")
st.sidebar.slider('W',-10,10,0,1)

class_ = getattr(nn,nl_txt)
NL = class_()

x = torch.randn(1000,2)
OI = torch.cat([torch.zeros(2,2),torch.eye(2)])



col1,col2 = st.beta_columns(2)

if visualize:
    colors = torch.randn(1000,1)
    
    W = torch.randn(2,2)
    xt = x@W.t()

    xnl = NL(xt)

    col1.write("W")
    col1.write(W.numpy())

    U,S,V = torch.svd(W)
    col2.write("U")
    col2.write(U.numpy())
    col2.write("V")
    col2.write(V.numpy())
    col2.write("S")
    col2.write(S.numpy())

    xd = pd.DataFrame({"x":x[:,0],
                        "y":x[:,1],
                        "xt":xt[:,0],
                        "yt":xt[:,1],
                        "xnl":xnl[:,0],
                        "ynl":xnl[:,1]})
    brush = alt.selection_interval()

    c = alt.Chart(xd).mark_circle().properties(width=300,height=300)\
        .add_selection(brush)

    cx = c.encode(alt.X('x',axis=None),alt.Y('y',axis=None),tooltip=['x','y'],color=alt.condition(brush,'x',alt.value('lightgray'),legend=None))
    cxt = c.encode(alt.X('xt',axis=None),alt.Y('yt',axis=None),tooltip=['xt','yt'],color=alt.condition(brush,'x',alt.value('lightgray')))
    cxnl = c.encode(alt.X('xnl',axis=None),alt.Y('ynl',axis=None),tooltip=['xnl','ynl'],color=alt.condition(brush,'x',alt.value('lightgray')))

    """
    ### X
    """

    st.write(alt.hconcat(cx,cxt,cxnl).configure_view(strokeWidth=0).configure_axis(grid=False))



# acts = [nn.ELU(), nn.Hardshrink(), nn.Hardsigmoid(), nn.Hardtanh(), nn.Hardswish(), nn.LeakyReLU(), nn.LogSigmoid(),\
#     nn.PReLU(), nn.ReLU(), nn.ReLU6(), nn.RReLU(), nn.SELU(), nn.CELU(), nn.GELU(), nn.Sigmoid(), nn.SiLU(), nn.Softplus(),\
#          nn.Softshrink(), nn.Softsign(), nn.Tanh(), nn.Tanhshrink()]


# plt.style.use(["dark_background","bmh"])
# plt.rc('axes', facecolor=(0,0,0,0))
# plt.rc('figure', facecolor=(0,0,0,0))
# plt.rc('figure', figsize=(40,40), dpi=100)

# for a in acts:
#     model = nn.Sequential(
#         nn.Linear(2,2,bias=False),
#         a
#     )

#     st.write(type(a).__name__)
#     fig,axs = plt.subplots(1,5)
#     #fig.suptitle(type(a).__name__,fontsize=25)
#     for i,ax in enumerate(axs):
#         W = (i+1)*torch.eye(2)
#         model[0].weight.data.copy_(W)
#         with torch.no_grad():
#             Y = model(x).data
#             plot_scatter(ax,Y)
#             plot_bases(ax,model(OI))
#     st.pyplot(fig)
