# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly
import plotly.graph_objs as go #visualization
import plotly.figure_factory as ff #visualization
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
init_notebook_mode()

from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
colors = plotly.colors.DEFAULT_PLOTLY_COLORS
churn_dict = {0: "no churn", 1: "churn"}


def missing_value_count(data):
    
    miss = data.isnull().sum()/len(data)
    miss = miss[miss > 0]
    miss.sort_values(inplace=True)

    #visualising missing values
    miss = miss.to_frame()
    miss.columns = ['count']
    miss.index.names = ['Name']
    miss['Name'] = miss.index

    #plot the missing value count
    sns.set(style="whitegrid", color_codes=True)
    sns.barplot(x = 'Name', y = 'count', data=miss)
    plt.xticks(rotation = 90)
    plt.title("Missing values count")
    plt.show()
    
def get_color_with_opacity(color, opacity):
    return "rgba(" + color[4:-1] + ", %.2f)" % opacity

def plot_feature_importance(feature_importance, title):
    
    trace1 = go.Bar(
        x=feature_importance[:, 0],
        y=feature_importance[:, 1],
        marker = dict(color = colors[0]),
        name='feature importance'
    )
    data = [trace1]
    layout = go.Layout(
        title=title,
        autosize=True,
        margin=go.layout.Margin(l=50, r=100, b=150),
        xaxis=dict(
            title='feature',
            tickangle=30
        ),
        yaxis=dict(
            title='feature importance',
            automargin=True,
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    return plot(fig, filename=title)

def pplot(sizes):
    
    explode = (0, .2)
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.pie(sizes, explode=explode, labels=['Retained','churned'], autopct='%1.1f%%',
        shadow=True, startangle=90)
    ax1.axis('equal')
    plt.title("Proportion of customer churned and retained", size = 20)
    plt.show()
    

    
def plot_pie(column,df_vis):
    churn     = df_vis[df_vis["churn"] == 1]
    not_churn = df_vis[df_vis["churn"] == 0]

    target_col = ["churn"]
    cat_cols   = df_vis.nunique()[df_vis.nunique() < 12].keys().tolist()
    cat_cols   = [x for x in cat_cols if x not in target_col]
    num_cols   = [x for x in df_vis.columns if x not in cat_cols + target_col]


    
    trace1 = go.Pie(values  = churn[column].value_counts().values.tolist(),
                    labels  = churn[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",domain  = dict(x = [0,.48]),
                    name    = "Churn Customers", marker  = dict(line = dict(width = 2,
                  color = "rgb(243,243,243)")),hole    = .6)
    
    trace2 = go.Pie(values  = not_churn[column].value_counts().values.tolist(),
                    labels  = not_churn[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name", marker  = dict(line = dict(width = 2,
                    color = "rgb(243,243,243)")),
                    domain  = dict(x = [.52,1]),
                    hole    = .6,
                    name    = "Non churn customers" )


    layout = go.Layout(dict(title = column + " distribution in customer database ",
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            annotations = [dict(text = "churn customers", font = dict(size = 13),
                                        showarrow = False, x = .15, y = .5),
                                           dict(text = "Non churn customers",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .80,y = .5
                                               )]))
    data = [trace1,trace2]
    fig  = go.Figure(data = data,layout = layout)
    plot(fig)


# Histogram distribution for churned data
#function  for histogram for customer churn types
def histogram(column,df_vis) :
    churn     = df_vis[df_vis["churn"] == 1]
    not_churn = df_vis[df_vis["churn"] == 0]

    target_col = ["churn"]
    cat_cols   = df_vis.nunique()[df_vis.nunique() < 12].keys().tolist()
    cat_cols   = [x for x in cat_cols if x not in target_col]
    num_cols   = [x for x in df_vis.columns if x not in cat_cols + target_col]


    trace1 = go.Histogram(x  = churn[column],
                          histnorm= "percent",
                          name = "Churn Customers",
                          marker = dict(line = dict(width = .5,
                                                    color = "black" )),opacity = .8 ) 
    trace2 = go.Histogram(x  = not_churn[column],
                          histnorm = "percent",
                          name = "Non churn customers",
                          marker = dict(line = dict(width = .5,
                         color = "black")), opacity = .9)
    
    data = [trace1,trace2]
    layout = go.Layout(dict(title =column + " distribution in the dataset ",
                            plot_bgcolor  = "rgb(225,245,248)",
                            paper_bgcolor = "rgb(243,243,243)",
                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = column,
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = "percent",
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                           )
                      )
    fig  = go.Figure(data=data,layout=layout)
    
    plot(fig)
    


def gen_boxplot(df):
    trace = []
    for feature in df:
        trace.append(
            go.Box(
                name = feature,
                y = df[feature]
            )
        )
    return trace


def plot_roc_curve(classifiers, legend, title, X_test, y_test):
    trace1 = go.Scatter(
        x=[0, 1], 
        y=[0, 1], 
        showlegend=False,
        mode="lines",
        name="",
        line = dict(
            color = colors[0],
        ),
    )
    
    data = [trace1]
    aucs = []
    for clf, string, c in zip(classifiers, legend, colors[1:]):
        y_test_roc = np.array([([0, 1] if y else [1, 0]) for y in y_test])
        y_score = clf.predict_proba(X_test)
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(y_test_roc[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_roc.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        aucs.append(roc_auc['micro'])

        trace = go.Scatter(
            x=fpr['micro'], 
            y=tpr['micro'], 
            showlegend=True,
            mode="lines",
            name=string + " (area = %0.2f)" % roc_auc['micro'],
            hoverlabel = dict(
                namelength=30
            ),
            line = dict(
                color = c,
            ),
        )
        data.append(trace)

    layout = go.Layout(
        title=title,
        autosize=False,
        width=550,
        height=550,
        yaxis=dict(
            title='True Positive Rate',
        ),
        xaxis=dict(
            title="False Positive Rate",
        ),
        legend=dict(
            x=0.4,
            y=0.06,
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    return aucs, plot(fig, filename=title)


def kdeplot(feature,df1):
    plt.figure(figsize=(9, 4.5))
    plt.title("KDE for {}".format(feature))
    ax0 = sns.kdeplot(df1[df1['churn'] == 0][feature], color= 'navy', label= 'Churn: No')
    ax1 = sns.kdeplot(df1[df1['churn'] == 1][feature], color= 'orange', label= 'Churn: Yes')