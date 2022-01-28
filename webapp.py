import streamlit as st
from pathlib import Path
import pandas as pd
from modules.utils.utils import read_json, build_connection_engine
from modules.utils.charting import plot_candlestick_with_signals_v2


p = Path(".")

config = read_json(p / "modules"/"config.json")
conn_s = build_connection_engine(config, 's')

table_schema = "cryptovaluer"
harmonized = "coin_harmonized_data"

df = pd.read_sql_query(f'select * from {table_schema}.{harmonized} where coin_id in (1,2,3)',con=conn_s)

st.write("""
# Heading
Cryptovaluer. 

### Cryptocoins technical signals

This is not financial advise!""")

st.write("""
## Testing PLotly charting
""")


coin_lst = list(df.coin_id.unique())
signal_cols = [col for col in df.columns if 'doji' in col]
ma_cols = [col for col in df.columns if any(s in col for s in ['ma_open', 'ma_close'])]

true_false = [True, False]


coin_mask = st.sidebar.selectbox('Select your coin:', coin_lst)

plot_signals = st.sidebar.selectbox('Do you want to plot a technical signal? :', true_false)
signal_mask = st.sidebar.selectbox('Select the technical signal you want to visualize:', signal_cols)
plot_trend = st.sidebar.selectbox('Do you want to plot a trend ? :', true_false)
ma_mask = st.sidebar.selectbox('Select the time window for the trend:', ma_cols)
plot_bollinger = st.sidebar.selectbox('Do you want to plot the Bollinger bands ? :', true_false)
bollinger_mask = st.sidebar.selectbox('Select the time window for the bollinger banda:', ma_cols)


#st.plotly_chart(fig, use_container_width=True)
filtered_df = df[df['coin_id']==coin_mask]

fig = plot_candlestick_with_signals_v2(filtered_df, signal_mask, ma_mask, plot_signal =plot_signals, plot_trend = plot_trend, bollinger_band = plot_bollinger)
st.plotly_chart(fig, use_container_width=False)