import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from modules.utils.utils import build_df, build_url, read_json, build_coin_list_table


url = build_url()
dct_ = build_df()
config = read_json("modules/config.json")

# get html page for historical
response = requests.get(url)

# beautifulSoup creation
soup = BeautifulSoup(response.text, "lxml")


# extract the information regarding each day
main_info = re.findall(
    config.get("HTML_EXTRACTION").get("main_pattern"), soup.find_all("p")[0].text
)

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

hist = pd.DataFrame.from_dict(dct_)
# print(df.head())

# get the list of coins, we want to refresh the db everytime we run the script

id_lst_active, id_lst_inactive, id_lst_untracked = (
    build_coin_list_table(config),
    build_coin_list_table(config, "inactive"),
    build_coin_list_table(config, "untracked"),
)

