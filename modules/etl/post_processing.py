from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from modules.utils.util import read_json, build_connection_engine, initialize_db_v2, load_to_postgres_v2, doji_star, calc_ma_slope_std


def run_harmonization(insert_new_signals = False):
    """
    This function runs the post processing harmonization of the extracted raw historical data. We calculate technical signals and trends. After the post processing, we store this information into the Postgres DB. It acts only on active coins

    :param insert_new_signals (bool) this specifies if we want to accomplish a drop of the table because we want to insert new column containing additional signals and features

    returns load post processed data to DB
    """
    # load requirements
    p = Path(".")

    config = read_json(p / "modules"/"config.json")
    #conn = build_connection_engine(config)
    #conn_s = build_connection_engine(config, 's')

    ## DB initialization
    table_schema = "cryptovaluer"
    listing_status = 'active'

    coin_lst_t = config.get("DATABASE").get("TABLES").get("coin_list_t")
    hist_coin_t = config.get("DATABASE").get("TABLES").get("historical_data_t")
    harmonized_t = config.get("DATABASE").get("TABLES").get("harmonized_data_t")
    conn_s = build_connection_engine(config, 's')

    coin_lst_df = pd.read_sql_query(f'select distinct coin_id from {table_schema}.{coin_lst_t} where listing_status like ' + "'" + str(listing_status) + "'",con=conn_s)
    coin_lst = list(coin_lst_df.coin_id)
    
    tmp_lst = []

    for ix, coin in enumerate(coin_lst):
        df = pd.read_sql_query(f'select * from {table_schema}.{hist_coin_t} where coin_id = {coin}',con=conn_s)
        df['date'] = pd.to_datetime(df['time_open']) # time open or time close is the same, as we have daily data
        # generate some additional features
        df['open_close_pct'] = np.abs((df.open_price - df.close_price)/df.open_price)
        df['mean_to_high_pct'] = np.abs((df.high_price - (df.open_price + df.close_price)/2) / df.high_price)
        df['mean_to_low_pct'] = np.abs((df.low_price - (df.open_price + df.close_price)/2) / df.low_price)
        df['open_close_mean'] = np.abs((df.open_price + df.close_price)/2)
        df['high_to_mean_distance_pct'] = (df.high_price - df.open_close_mean)/df.high_price
        df['mean_to_low_distance_pct'] = (df.open_close_mean - df.low_price)/df.open_close_mean
        df['high_to_mean_distance_pct_magnitude'] = df['high_to_mean_distance_pct'].apply(lambda x: np.floor(np.log10(x)))
        df['mean_to_low_distance_pct_magnitude'] = df['mean_to_low_distance_pct'].apply(lambda x: np.floor(np.log10(x)))

        # signal heuristic for technical analysis
        df['doji'] = np.where(
            # rule to identify doji if open close pct is at or lower 2% and mean to high minus mean to low are ~ at 1% then is a doji
            (df.open_close_pct <= 0.005) &  ((np.abs(df.mean_to_high_pct - df.mean_to_low_pct)	 <= 0.005)), 1, 0
        )

        df['dragonfly'] = np.where(
            # rule to identify dragonfly is to check that open and close price are the same (only 0.5% difference) and the magnitude of the tail is towards the lower price
            (df.open_close_pct <= 0.005) &  (np.abs(df.mean_to_low_distance_pct.values) - np.abs(df.high_to_mean_distance_pct_magnitude.values) <= 1) & (df.doji == 0) & (df.mean_to_low_pct - df.mean_to_high_pct > 0.02)
            , 1, 0
        )

        df['gravestone'] = np.where(
            # rule to identify dragonfly is to check that open and close price are the same (only 0.5% difference) and the magnitude of the tail is towards the higher price
            (df.open_close_pct <= 0.005) &  (np.abs(df.mean_to_low_distance_pct.values) - np.abs(df.high_to_mean_distance_pct_magnitude.values) <= 1)  & (df.mean_to_high_pct - df.mean_to_low_pct > 0.02) & ((df.doji == 0) & (df.dragonfly == 0))
            , 1, 0
        )

        # generate the MA 
        cols = ['open_price', 'close_price', 'high_price', 'low_price']
        time_windows = [5, 10, 20, 50]

        for col in cols:
            for t in time_windows:
                df = calc_ma_slope_std(df, col, t)
                df = doji_star(df, col, t, t)

        df['close_price_to_ma_20_ratio'] = df['close_price']/df['ma_close_price_20'] 
        df['open_price_to_ma_20_ratio'] = df['open_price']/df['ma_open_price_20'] 
        
        
        df['updated_at'] = datetime.now()
        tmp_df = df[df.columns[~df.columns.isin(['created_at','updated_at'])]]
        tmp_df['created_at'] = df['created_at']
        tmp_df['updated_at'] = df['updated_at']

        tmp_lst.append(tmp_df)
        print(f"Post processing for coin {coin} complete. Processing next coin ...")

    df = pd.concat(tmp_lst)
    # initialize Postgres DB table
    initialize_db_v2(config, df, harmonized_t, table_schema, insert_new_signals )
    # load the new df into Postgres Harmonization table
    load_to_postgres_v2(config, df, harmonized_t, table_schema)
    
    return print("Post processing completed")