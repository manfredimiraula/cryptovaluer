
from turtle import color
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def plot_candlestick_with_signals_v2(df, signal_col, trend_col, start_date = None, end_date = None, plot_signal = False, plot_trend = False, bollinger_band = False):

    if start_date:
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # include candlestick with rangeselector
    fig.add_trace(
        go.Candlestick(x=df['date'],
                    open=df['open_price'],
                    high=df['high_price'],
                    low=df['low_price'],
                    close=df['close_price']
                    ), 
                secondary_y=True
                )
    if plot_signal:
        addTechnicalSignalBars(fig, df, signal_col)
    if plot_trend:
        addTrend(fig, df, trend_col)
    # include a go.Bar trace for volumes
    
    if bollinger_band:
        addMA(fig, df, trend_col)
        addUpperBand(fig, df, trend_col)
        addLowerBand(fig, df, trend_col)

    fig.layout.yaxis2.showgrid=False
    fig.update_layout(
        autosize=True,
        width=900,
        height=600,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=1
        ),
         paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        #paper_bgcolor="LightSteelBlue",
        legend=dict(
            orientation = 'h',
            yanchor="top",
            y=1.15,
            xanchor="center",
            x=0.5, 
            title_font_family="Courier",
            font=dict(
                family="Courier",
                size=16,
                color="white"
            )
        ), 
       
    )
    return fig

    return fig



def getArgsByButton(df, coin):
        """
        This funtcion select the args to pass to the plot. 
        It creates the dict with the data information
        """
        x = list(df[df['coin_id'] == coin]['date']),
        open =list(df[df['coin_id'] == coin]['open_price']),
        high = list(df[df['coin_id'] == coin]['high_price']),
        low = list(df[df['coin_id'] == coin]['low_price']),
        close = list(df[df['coin_id'] == coin]['close_price']),
        label = df[df['coin_id'] == coin]['name'].unique()[0]
        
        return [{'x':x, 'open':open, 'high':high, 'low':low, 'close':close}]

def getLableByButton(df, coin):
    """ 
    This creates the label for the specific list
    """
    return df[df['coin_id'] == coin]['name'].unique()[0]

def addTechnicalSignalBars(fig, df, signal_col):
    """This function add a Plotly trace with a specific technical signal"""
    fig = fig.add_trace(
        go.Bar(
            x=df[(df[signal_col]==1)].date, 
            y=[df['high_price'].max()], 
            name = f"Plotting the signal {signal_col}",
            marker=dict(
                    color = list(['black'] * len(df[(df[signal_col]==1)]['date'])))
            ),
            secondary_y=False, 
            )
    return fig

def addTrend(fig, df, trend_col = 'ma_close_price_20'):
    fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df[trend_col],
                name = f"Trend {trend_col}"
                ),
                    secondary_y=False
                    )

def addMA(fig, df, trend_col = 'ma_close_price_20'):
    return fig.add_trace(
            go.Scatter(
                x=df.date, 
                y=df[trend_col],
                mode='lines',
                name=trend_col,
                line = dict(
                    color='#1E90FF', 
                    width=2, 
                )
            )
        )

def addUpperBand(fig, df, trend_col = 'ma_close_price_20'):
    return fig.add_trace(
                go.Scatter(
                    x=df.date, 
                    y=(df[trend_col] + df[f"std_{trend_col[3:]}"]),
                    mode='lines',
                    name='upper band',
                    line = dict(
                        color='#1E90FF', 
                        width=0.5, 
                    )
                )
            )
def addLowerBand(fig, df, trend_col = 'ma_close_price_20'):
    return fig.add_trace(
                go.Scatter(
                    x=df.date, 
                    y=(df[trend_col] - df[f"std_{trend_col[3:]}"]),
                    mode='lines',
                    name='lower band',
                    fill='tonexty',
                    line = dict(
                        color='#1E90FF', 
                        width=0.5, 
                    )
                )
            )


def plot_candlestick_with_signals(df, signal_col, trend_col, start_date = None, end_date = None, plot_signal = False, plot_trend = False, bollinger_band = False):
    """
    This function plots a plotly candlestick added with external signals and or trends

    :param df
    :param signal_col
    :param: trend_col
    :param start_date and end_date
    :param plot_signal (bool)
    :param plot_trend (cool)
    :param bollinger_band(bool)

    returns plotly plot
    """

    if start_date:
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    coin_lst = list(df['coin_id'].unique())

    button_lst = []
    for coin in coin_lst:
        button_lst.append(dict(
                        args= getArgsByButton(df, coin),
                        label=getLableByButton(df, coin),
                        method="update"
                    ))


    # include candlestick with rangeselector
    fig.add_trace(
        # starting plot is Bitcoin
        go.Candlestick(x=df[df['coin_id']==1]['date'],
                    open=df[df['coin_id']==1]['open_price'],
                    high=df[df['coin_id']==1]['high_price'],
                    low=df[df['coin_id']==1]['low_price'],
                    close=df[df['coin_id']==1]['close_price']
                    ), 
                secondary_y=True
                )
    
    if plot_signal:
        addTechnicalSignalBars(fig, df, coin, signal_col)
    # include a go.Bar trace for volumes
    

    if plot_trend:
        addTrend(fig, df, coin, trend_col)
    # include a go.Bar trace for volumes
        
    if bollinger_band:
        addMA(fig, df, coin, trend_col)
        addUpperBand(fig, df, coin, trend_col)
        addLowerBand(fig, df, coin, trend_col)

    fig.layout.yaxis2.showgrid=False
    fig.update_layout(
        autosize=True,
        width=1800,
        height=400,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=1
        ),
        #paper_bgcolor='rgba(0,0,0,0)',
        #plot_bgcolor='rgba(0,0,0,0)',
        #paper_bgcolor="LightSteelBlue",
        legend=dict(
            orientation = 'h',
            yanchor="top",
            y=1.15,
            xanchor="center",
            x=0.5, 
            title_font_family="Times New Roman",
            font=dict(
                family="Courier",
                size=12,
                color="white"
            )
        ), 
        updatemenus=[
        dict(
            buttons=list(button_lst),
            direction="down",
            #pad={"r": 0, "t": 0},
            showactive=True,
            x=-0.01,
            #xanchor="left",
            #y=1.1,
            yanchor="top"
        ),
    ]
    )
    return fig
