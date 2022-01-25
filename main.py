

from modules.etl.etl import run_etl
from modules.etl.post_processing import run_harmonization


# run the ETL pipeline that 1) initialize the DB, 2) extract the coin list information, 3) load the coin list to DB, 4) extract the historical information for the coins, 5) load this information into the DB

#run_etl()

run_harmonization(insert_new_signals = False)

