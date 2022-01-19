from pathlib import Path
from numpy import float64
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from modules.utils.utils import build_df, build_url, read_json, build_coin_list_table, initialize_db, build_connection_engine, read_coin_list_from_db, load_to_postgres, extracted_data_to_df
from sqlalchemy import text

def run_etl():
    """
    This functions wrap the ETL process. It is called by the main.py and it is self exhaustive. There are no parameters needed to initialize the function.
    """
    # load requirements
    p = Path(".")

    config = read_json(p / "modules"/"config.json")
    #conn = build_connection_engine(config)
    #conn_s = build_connection_engine(config, 's')

    ## DB initialization
    table_schema = "cryptovaluer"
    coin_lst_t = config.get("DATABASE").get("TABLES").get("coin_list_t")
    hist_coin_t = config.get("DATABASE").get("TABLES").get("historical_data_t")

    initialize_db(config, coin_lst_t, table_schema)
    initialize_db(config, hist_coin_t, table_schema)

    # get the list of coins, we want to refresh the db everytime we run the script
    id_lst_active, id_lst_inactive, id_lst_untracked = (
        build_coin_list_table(config),
        build_coin_list_table(config, "inactive"),
        build_coin_list_table(config, "untracked"),
    )
    # insert/append list of coins
    load_to_postgres(config, id_lst_active, coin_lst_t, table_schema)
    load_to_postgres(config, id_lst_inactive, coin_lst_t, table_schema)
    load_to_postgres(config, id_lst_untracked, coin_lst_t, table_schema)

    # read the coin list and 
    coin_lst = read_coin_list_from_db(config, coin_lst_t, table_schema)
    #coin_lst = coin_lst[coin_lst['coin_id'] == 184]

    # extract the historical data regarding all the coins
    for row, coin in coin_lst.iterrows():
        url = build_url(coin['coin_id'], "eur")    
        print(f"Loading coin {coin['coin_id']}, with name {coin['name']}, symbol:{coin['symbol']}")
        
        # get html page for historical
        response = requests.get(url)

        # beautifulSoup creation
        soup = BeautifulSoup(response.text, "lxml")

        # extract the information regarding each day
        main_info = re.findall(
            config.get("HTML_EXTRACTION").get("main_pattern"), soup.find_all("p")[0].text
        )
        dct_ = build_df()
        # load the information into pandas DF
        hist = extracted_data_to_df(main_info, dct_, coin)
        # load to Postgres 
        load_to_postgres(config, hist, hist_coin_t, table_schema)


        