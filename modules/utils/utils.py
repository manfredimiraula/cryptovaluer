import json
import time
from time import sleep
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import pandas as pd


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

    return df
