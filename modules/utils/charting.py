
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def plot_candlestick_with_signals(df, signal_col, trend_col, start_year = None, year_end = None, plot_signal = False, plot_trend = False, bollinger_band = False):
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
    
    if start_year is not None:
        df = df[(pd.DatetimeIndex(df['date']).year >= start_year)]
    elif year_end is not None:
        df = df[(pd.DatetimeIndex(df['date']).year <= year_end)]
    elif start_year is not None and year_end is not None:
        df = df[ (pd.DatetimeIndex(df['date']).year >= start_year) & (pd.DatetimeIndex(df['date']).year <= year_end)]
    else:
        df

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # include candlestick with rangeselector
    fig.add_trace(
        # starting plot is Bitcoin
        go.Candlestick(x=df['date'],
                    open=df['open_price'],
                    high=df['high_price'],
                    low=df['low_price'],
                    close=df['close_price']
                    ), 
                secondary_y=True
                )

    # adding technical signals. This can be a func
    if plot_signal:
        fig = addTechnicalSignal(fig, df, signal_col)

    if plot_trend:
        fig = addTrend(fig, df, trend_col)

    if bollinger_band:
        addBollingerBands(fig, df, trend_col)


    fig.layout.yaxis2.showgrid=False
    fig.update_layout(
        autosize=False,
        width=800,
        height=600,

        paper_bgcolor="LightSteelBlue",
    )
    

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

def addTechnicalSignal(df, signal_col):
    fig = fig.add_trace(
            go.Scatter(
                x=df[(df[signal_col]==1)].date, 
                y=df[(df[signal_col]==1)].high_price.values + 30 ,
                name = f"Plotting the signal {signal_col}",
                mode = "markers",
                marker=dict(
                    color='LightSkyBlue', # maybe change the colour
                    size=1,
                    line=dict(
                        color='DarkSlateGrey',
                        width=2
                )
            ),
        ),
        secondary_y=True, 
        )
    return fig

def getLableByButton(df, coin):
    """ 
    This creates the label for the specific list
    """
    return df[df['coin_id'] == coin]['name'].unique()[0]


def addTechnicalSignal(fig, df, signal_col):
    fig = fig.add_trace(
            go.Scatter(
                x=df[(df[signal_col]==1)].date, 
                y=df[(df[signal_col]==1)].high_price.values + 30 ,
                name = f"Plotting the signal {signal_col}",
                mode = "markers",
                marker=dict(
                    color='LightSkyBlue', # maybe change the colour
                    size=1,
                    line=dict(
                        color='DarkSlateGrey',
                        width=2
                    )
                ),
            ),
    secondary_y=True, 
    )

    return fig

def addTrend(fig, df, trend_col):
    fig.add_trace(
        go.Scatter(
                x=df['date'],
            y=df[trend_col],
            name = f"Trend {trend_col}"
        ),
    secondary_y=False
    )
    return fig

def addBollingerBands(fig, df, trend_col):
    fig.add_trace(
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

    fig.add_trace(
            go.Scatter(
                x=df.date, 
                y=(df[trend_col] + df[trend_col.replace('ma', 'std')]),
                mode='lines',
                name='upper band',
                line = dict(
                    color='#1E90FF', 
                    width=0.5, 
                )
            )
        )

    fig.add_trace(
            go.Scatter(
                x=df.date, 
                y=(df[trend_col] - df[trend_col.replace('ma', 'std')]),
                mode='lines',
                name='lower band',
                fill='tonexty',
                line = dict(
                    color='#1E90FF', 
                    width=0.5, 
                )
            )
        )
    return fig

