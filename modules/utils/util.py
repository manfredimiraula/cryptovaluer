import json
import time
from time import sleep
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import pandas as pd
import numpy as np
# Postgres DB
from sqlalchemy import create_engine,  Integer, Numeric, Text
import psycopg2
from sqlalchemy.sql.expression import column


def read_json(file_path):
    """
    Read json config file

    param: file path of the .json

    returns the loaded json in the form of a python dictionary
    """
    with open(file_path, "r") as f:
        return json.load(f)


# current time in unix timestamp
unix_current_date = int(time.time())

# build url based on params
def build_url(coin="1", currency_id="eur"):
    """
    This function gets all the parameters from the config file and builds the url to scrape based on some inputs

    :params coin -> the coin id from coinMarketCap ID list. It is better to use the id as the name could be misleading or overlapping
    :param -> currency_id -> the default currency is eur
    :param -> unix start time is the starting time for all scraping process. Always 2000-01-01 as the earliest date
    :param -> unix current date is the current date in unix time

    returns str -> url to scrape
    """
    config = read_json("modules/config.json")
    # coin = config.get("coinMarketCap").get("coin_id").get(coin)
    currency_id = config.get("coinMarketCap").get("currency_id").get(currency_id)
    unix_start_time = config.get("coinMarketCap").get("unix_start_time")

    return f"https://api.coinmarketcap.com/data-api/v3/cryptocurrency/historical?id={coin}&convertId={currency_id}&timeStart={unix_start_time}&timeEnd={unix_current_date}"


def build_df():
    """
    Set up the dictionary to welcome the historical dataset

    returns a py dict
    """
    # build the structure to contain the extracted data
    dct_ = {
        "timeOpen": [],
        "timeClose": [],
        "timeHigh": [],
        "timeLow": [],
        "openPrice": [],
        "highPrice": [],
        "lowPrice": [],
        "closePrice": [],
        "volume": [],
        "mrktCap": [],
    }

    return dct_


def get_coin_id_list(config, listing_status=None):
    """
    Gets the response from coinMarketCap API and generate the list of coin ids with auxiliary information

    param: -> config.json
    returns a pandas df
    """

    url = config.get("coinMarketCap").get("coin_id").get("url")
    parameters = config.get("coinMarketCap").get("coin_id").get("parameters")
    headers = config.get("coinMarketCap").get("coin_id").get("headers")

    if listing_status is not None:
        parameters["listing_status"] = listing_status

    session = Session()
    session.headers.update(headers)

    try:
        response = session.get(url, params=parameters)
        data = json.loads(response.text)
        print("Success, the list of coin ids has been retrieved")
        return data
    except (ConnectionError, Timeout, TooManyRedirects) as e:
        print(e)


def build_coin_list_table(config, listing_status=None):
    """
    Gets the list of coin ids and auxilliary info from CMC API and returns a Pandas Dataframe

    param: conifg file 
    param: listing_status; default = active, else (str) inactive, untracked

    returns pandas dataframe
    """
    tmp = get_coin_id_list(config, listing_status)["data"]

    df = pd.DataFrame.from_dict(tmp)
    if listing_status is None:
        df['listing_status'] = "active"
    else:    
        df['listing_status'] = listing_status

    return df

def build_connection_engine(config, type = 'p'):
    """
    Create the PostgresDB connection using config and return the connection object. Based on type we create a psycopg2 engine or SQLAlchemy engine 

    :param -> username
    :param -> passowrd
    :param -> host
    :param -> port
    :param -> db

    return psycopg2 connection object
    """
    username = config.get("DATABASE").get("POSTGRES").get("USERNAME")
    password = config.get("DATABASE").get("POSTGRES").get("PASSWORD")
    host = config.get("DATABASE").get("POSTGRES").get("HOST")
    port = config.get("DATABASE").get("POSTGRES").get("PORT")
    db = config.get("DATABASE").get("POSTGRES").get("DB")
    if type == 'p':
        conn = psycopg2.connect(database=db, user=username,
                                password=password, host=host, port=port)
    else:
        conn = create_engine('postgresql+psycopg2://{}:{}@{}:{}/postgres'
                           .format(username,
                                   password,
                                   host,
                                   port
                                   ), echo=False)
    return conn

def get_column_dtypes(dtypes):
        dtype_lst = []
        for x in dtypes:
            if(x == 'int64'):
                dtype_lst.append('int')
            elif (x == 'float64'):
                dtype_lst.append('float')
            elif (x == 'bool'):
                dtype_lst.append('boolean')
            elif x == 'O':
                dtype_lst.append('text')
            elif x == '<M[ns]':
                dtype_lst.append('date')
            else:
                dtype_lst.append('date')
        return dtype_lst

def build_table_structure_v2(df, schema, table):
    """
    Build the table structure based on the table name. It automatically builds the SQL table structure with correct datatypes by inferring from the DataFrame.

    :param df -> panda DataFrame containing the dat

    returns the table structure based on the df
    """ 
    col_name = list(df.columns)
    
    col_dtypes = get_column_dtypes(df.dtypes)

    create_statement = f'CREATE TABLE IF NOT EXISTS {schema}.{table} ('
    for i in range(len(col_dtypes)):
        create_statement = create_statement + '\n' + col_name[i] + ' ' + col_dtypes[i] + ', '
    create_statement = create_statement[:-2]  + ')'     
    index_keys = """(coin_id, name)"""
    return create_statement, index_keys

def initialize_db_v2(config, df, table_name, table_schema, insert_new_signals = False):
    """
    Function that creates the connection with PostgresDB and initialize the table needed if not present. The table is declared in the appropriate function and passed here.


    :param -> table_name (str) the name of the table
    :param -> table_schema (str) the schema of the table
    :param -> engine the connection engine
    :param -> conn the connection to the DB

    return 
    """
    conn = build_connection_engine(config)
    
    cur = conn.cursor()
    if insert_new_signals:
        cur.execute(
            f"DROP TABLE IF EXISTS {table_schema}.{table_name}"
        )
        print(f"Table {table_name} has been dropped")
    cur.execute(
        "select * from information_schema.tables where table_name=%s and table_schema = %s", (table_name, table_schema))
    check = bool(cur.rowcount)

    if check == False:
        print("Initializing table")
        table_structure, index_keys = build_table_structure_v2(df, table_schema, table_name)
        #create schema if doesn't exist
        cur.execute(
            f"""CREATE SCHEMA IF NOT EXISTS {table_schema} AUTHORIZATION {config.get("DATABASE").get("POSTGRES").get("USERNAME")};"""
            )
        # initialize table if doesn't exist
        cur.execute(
            f"""
                DROP TABLE if EXISTS {table_schema}.{table_name};
                 {table_structure}
                WITH (
                        OIDS = FALSE
                        )
                        TABLESPACE pg_default;
                        ALTER TABLE {table_schema}.{table_name}
                        OWNER to manfredi;
                        CREATE INDEX {table_name}_pkid ON {table_schema}.{table_name}{index_keys};
                """
        )

        conn.commit()  # <--- makes sure the change is shown in the database
        conn.close()
        cur.close()
    else:
        print('The database is already initialized')


def build_table_structure(table_name):
    """
    Build the table structure needed for the specific tables
    """
    if table_name == "coin_list":
        table_structure = """(
                        coin_id int, 
                        name text, 
                        symbol text,
                        slug text,
                        rank int,
                        is_active int, 
                        first_historical_data text, 
                        last_historical_data text,
                        platform text,
                        listing_status text, 
                        created_at timestamp without time zone NOT NULL DEFAULT NOW(),
                        updated_at timestamp without time zone DEFAULT NULL)
        """
        index_keys = """(coin_id, name, rank)"""
    elif table_name == "historical_coin_data":
        table_structure = """(
                        coin_id int,
                        name text,
                        rank int,
                        time_open date, 
                        time_close date, 
                        time_high date, 
                        time_low date, 
                        open_price float, 
                        close_price float,
                        high_price float,
                        low_price float, 
                        vol float, 
                        mrkt_cap float, 
                        created_at timestamp without time zone NOT NULL DEFAULT NOW(),
                        updated_at timestamp without time zone DEFAULT NULL)
        """
        index_keys = """(coin_id, name, rank)"""
    elif table_name == "coin_harmonized_data":
        table_structure = """(
                        coin_id int,
                        name text,
                        rank int,
                        time_open date, 
                        time_close date, 
                        time_high date, 
                        time_low date, 
                        open_price float, 
                        close_price float,
                        high_price float,
                        low_price float, 
                        vol float, 
                        mrkt_cap float, 
                        created_at timestamp without time zone NOT NULL DEFAULT NOW(),
                        updated_at timestamp without time zone DEFAULT NULL)
        """
        index_keys = """(coin_id, name, rank)"""

    return table_structure, index_keys

def read_coin_list_from_db(config, table_name, table_schema):
    """
    Extract the list of coins from the DB and create the list to extract the historical information.

    :param -> config.json

    return a df of coin id (key) and name,symbol (lst)
    """
    conn = build_connection_engine(config)
    
    sql = f"select * from {table_schema}.{table_name};"
    df = pd.read_sql_query(sql, conn)
    conn = None

    return df
        

def initialize_db(config, table_name, table_schema):
    """
    Function that creates the connection with PostgresDB and initialize the table needed if not present. The table is declared in the appropriate function and passed here.


    :param -> table_name (str) the name of the table
    :param -> table_schema (str) the schema of the table
    :param -> engine the connection engine
    :param -> conn the connection to the DB

    return 
    """
    conn = build_connection_engine(config)
    
    cur = conn.cursor()

    cur.execute(
        "select * from information_schema.tables where table_name=%s and table_schema = %s", (table_name, table_schema))
    check = bool(cur.rowcount)

    if check == False:
        print("Initializing table")
        table_structure, index_keys = build_table_structure(table_name)
        #create schema if doesn't exist
        cur.execute(
            f"""CREATE SCHEMA IF NOT EXISTS {table_schema} AUTHORIZATION {config.get("DATABASE").get("POSTGRES").get("USERNAME")};"""
            )
        # initialize table if doesn't exist
        cur.execute(
            f"""
                DROP TABLE if EXISTS {table_schema}.{table_name};
                CREATE TABLE IF NOT EXISTS {table_schema}.{table_name} {
                    table_structure
                }
                WITH (
                        OIDS = FALSE
                        )
                        TABLESPACE pg_default;
                        ALTER TABLE {table_schema}.{table_name}
                        OWNER to manfredi;
                        CREATE INDEX {table_name}_pkid ON {table_schema}.{table_name}{index_keys};
                """
        )

        conn.commit()  # <--- makes sure the change is shown in the database
        conn.close()
        cur.close()
    else:
        print('The database is already initialized')


def load_to_postgres(config, df, table_name, table_schema):
    """
    This function will upload the data extracted from the MCM page from a single coin history to the table. 
    This is an iterative process, thus we will check the latest data available and append new data

    :param 

    returns
    """
    # create the list of entries that are already present at the db
    conn = build_connection_engine(config)
    cur = conn.cursor()

    # list for coins
    if table_name == 'coin_list':
        cur.execute(
            "select coin_id, last_historical_data from cryptovaluer.coin_list")

        tmp_df = pd.DataFrame(cur.fetchall(), columns=['coin_id', 'last_historical_data'])
        coin_id_lst = list(tmp_df.coin_id.unique())
        latest_hist = list(tmp_df.last_historical_data.unique())

        df['platform'] = df['platform'].astype(str)
        df.rename(columns = {'id': 'coin_id'}, inplace = True)   
        
        try:
            df = df[~df['coin_id'].isin(coin_id_lst) & ~df['last_historical_data'].isin(latest_hist)]
        except:
            print('There is no match in the DB')
    elif table_name == 'historical_coin_data':
        cur.execute(
            "select coin_id, max(time_open) as time_open from cryptovaluer.historical_coin_data group by coin_id")

        tmp_df = pd.DataFrame(cur.fetchall(), columns=['coin_id', 'time_open'])
        coin_id_lst = list(tmp_df.coin_id.unique())
        latest_time = list(tmp_df.time_open.unique())
        
        try:
            df = df[~df['coin_id'].isin(coin_id_lst) & ~df['time_open'].isin(latest_time)]
        except:
            print('There is no match in the DB')
    
    # we load into Postgres table created
    
    conn = build_connection_engine(config, 's')
    df.to_sql(name=table_name,
               schema=table_schema,
               con=conn,
               if_exists='append',
               index=False,
               chunksize=1000,
               method='multi',
    )

    print('Inserted '+str(len(df))+' rows ' + "in " +
          str(table_schema) + "." + str(table_name))

def load_to_postgres_v2(config, df, table_name, table_schema):
    """
    This function will upload the data extracted from the MCM page from a single coin history to the table. 
    This is an iterative process, thus we will check the latest data available and append new data

    :param 

    returns
    """
    # create the list of entries that are already present at the db
    conn = build_connection_engine(config)
    cur = conn.cursor()
    coin_id = 'coin_id'
    last_coin_data = 'last_historical_data'
    time_open = 'time_open'
    updated_at = 'updated_at'

    # list for coins
    if table_name == 'coin_list':
        cur.execute(
            f"select {coin_id}, {last_coin_data} from {table_schema}.{table_name}")
        tmp_df = pd.DataFrame(cur.fetchall(), columns=[{coin_id}, {last_coin_data}]
        )
        # specific for coin list table
        coin_id_lst = list(tmp_df.coin_id.unique())
        latest_hist = list(tmp_df.last_historical_data.unique())

        df['platform'] = df['platform'].astype(str)
        df.rename(columns = {'id': 'coin_id'}, inplace = True)       
        try:
            df = df[~df[coin_id].isin(coin_id_lst) & ~df[last_coin_data].isin(latest_hist)]
        except:
            print('There is no match in the DB')
    
    elif table_name == 'historical_coin_data':
        cur.execute(
            f"select {coin_id}, max({time_open}) as {time_open} from {table_schema}.{table_name} group by {coin_id}")

        tmp_df = pd.DataFrame(cur.fetchall(), columns=[coin_id, time_open])
        coin_id_lst = list(tmp_df.coin_id.unique())
        latest_time = list(tmp_df.time_open.unique())
        try:
            df = df[~df[coin_id].isin(coin_id_lst) & ~df[time_open].isin(latest_time)]
        except:
            print('There is no match in the DB')
    
    elif table_name == 'coin_harmonized_data':
        cur.execute(
            f"select {coin_id}, max({updated_at}) as {updated_at} from {table_schema}.{table_name} group by {coin_id}")

        tmp_df = pd.DataFrame(cur.fetchall(), columns=[coin_id, updated_at])
        coin_id_lst = list(tmp_df.coin_id.unique())
        latest_time = list(tmp_df.updated_at.unique())
        try:
            df = df[~df[coin_id].isin(coin_id_lst) & ~df[time_open].isin(latest_time)]
        except:
            print('There is no match in the DB')
    # we load into Postgres table created
    
    conn = build_connection_engine(config, 's')
    df.to_sql(name=table_name,
               schema=table_schema,
               con=conn,
               if_exists='append',
               index=False,
               chunksize=1000,
               method='multi',
    )

    print('Inserted '+str(len(df))+' rows ' + "in " +
          str(table_schema) + "." + str(table_name))

def extracted_data_to_df(main_info, dct_, coin):
    """
    This function generates the final pandas DataFrame that will be loaded to Postgres. We store the information contained in the HTML in the final format for the DB table

    :param main_info -> html_extracted data
    :param dct_
    :param coin -> this is the row from the coin_list for a specific coin

    returns pandas DataFrame
    """
    for date, hist in enumerate(main_info):
        # date list pos, hist info of single event
        # optimize text for info extraction
        tmp = hist.replace('","', ",")
        tmp = tmp.replace('":"', ",")
        tmp = tmp.replace('{"', ",")
        tmp = tmp.replace('":', ",")
        tmp = tmp.replace(',"', ",")
        tmp = tmp.replace(",,", ",")

        # split the content on the comas
        tmp_split = tmp.split(",")

        for i, val in enumerate(tmp_split):
            if i == 1:
                dct_["timeOpen"].append(val.replace("''", ""))
            elif i == 3:
                dct_["timeClose"].append(val.replace("''", ""))
            elif i == 5:
                dct_["timeHigh"].append(val.replace("''", ""))
            elif i == 7:
                dct_["timeLow"].append(val.replace("''", ""))
            elif i == 10:
                dct_["openPrice"].append(val.replace("''", ""))
            elif i == 12:
                dct_["highPrice"].append(val.replace("''", ""))
            elif i == 14:
                dct_["lowPrice"].append(val.replace("''", ""))
            elif i == 16:
                dct_["closePrice"].append(val.replace("''", ""))
            elif i == 18:
                dct_["volume"].append(val.replace("''", ""))
            elif i == 20:
                dct_["mrktCap"].append(val.replace("''", ""))

    hist = pd.DataFrame(dict([(col_name,pd.Series(values)) for col_name,values in dct_.items() ]))
    date_cols = [
        'timeOpen', 'timeClose', 'timeHigh', 'timeLow' 
        ]
    float_cols = [
        'openPrice', 'closePrice', 'highPrice', 'lowPrice', 'volume', 'mrktCap'
    ]
    hist[date_cols] = hist[date_cols].apply(lambda x: pd.to_datetime(x,errors='coerce'))
    hist[float_cols] = hist[float_cols].astype(float)
    hist['coin_id'] = coin['coin_id']
    hist['name'] = coin['name']
    hist['rank'] = coin['rank']
    hist = hist[[
        'coin_id',
        'name', 
        'rank', 
        'timeOpen', 
        'timeClose', 
        'timeHigh', 
        'timeLow', 
        'openPrice', 
        'closePrice', 
        'highPrice', 
        'lowPrice', 
        'volume', 
        'mrktCap'
        ]]
    hist.rename(columns={
        'timeOpen':'time_open', 
        'timeClose':'time_close', 
        'timeHigh':'time_high', 
        'timeLow':'time_low', 
        'openPrice':'open_price', 
        'closePrice':'close_price', 
        'highPrice':'high_price', 
        'lowPrice':'low_price', 
        'volume':'vol', 
        'mrktCap':'mrkt_cap'
    }, inplace=True)

    return hist
    


    ########## Post processing helper functions
def assign_signal_label(df, date, col, val):
    """Based on a specific date, assign the boolean label for a specific technical signal specified in a column
    :param df
    :param date
    :param col
    :param val

    returns the dataframe with the reassigned row
    """
    df.loc[df.date == date, col] = val

    return df

def orderOfMagnitude(number):
    """"""

    return math.floor(math.log(number, 10))

def calc_ma_slope_std(df, col, time_window):
    """
    This function calculates the moving average and the slope of a set number of rows based on a defined time window

    :param df -> pandas dataframe
    :param col -> the name of the column to be considered for the ma and slope
    : time_window (int) -> the number of days that will be used to set the window for ma and slope

    returns a pandas DataFrame
    """

    df[f'ma_{col}_{time_window}'] = df[col].rolling(window=time_window).mean()
    df[f'std_{col}_{time_window}'] = df[col].rolling(window=time_window).std()
    df[f'slope_{col}_{time_window}'] = df[col].rolling(time_window, min_periods = time_window).apply(calc_slope)

    return df

def calc_slope(x):
    """
    Simple slope calulation
    """
    slope = np.polyfit(range(len(x)), x, 1)[0]
    return slope
    
def doji_star(df, col, slope, time_window):
    """
    This function label a row in a pandas dataframe based on a predefined set of constraints. 
    In particular, we obtain a morning doji star if the point is already labelled as a doji and if there is a downward trend reversal. Vice versa, an evening doji star is obtained if there is an upward trend reversal. The trend reversal is calculated based on the mean value of the slope in a predefined time window previous to the current point. The time window used can be different from the slope used to evaluate the trend

    :param df -> pandas DataFrame
    :param col -> the column name of the values to be used
    :param slope -> the type of slope from 5, 10, 20, 50
    :param time_window -> the time window to be used to calculate the mean of the slope through the rolling function. 

    Returns a pandas DataFrame
    """
    df[f'morning_doji_star_{time_window}'] = np.where(
        # here we want to check if there is a doji and there is a trend reversal. to check if there is a trend reversal
        # we impose to check wether the slope of a set time window is changing sign
        (df.doji == 1) & (df[f'slope_{col}_{slope}'] < 0) & ( df[f'slope_{col}_{slope}'].rolling(time_window).mean() > 0), 1, 0
    )

    df[f'evening_doji_star_{time_window}'] = np.where(
        # here we want to check if there is a doji and there is a trend reversal. to check if there is a trend reversal
        # we impose to check wether the slope of a set time window is changing sign
        (df.doji == 1) & (df[f'slope_{col}_{slope}'] > 0) & ( df[f'slope_{col}_{slope}'].rolling(time_window).mean() < 0), 1, 0
    )

    return df
