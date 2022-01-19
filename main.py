
from modules.etl.etl import run_etl
import streamlit as st

# run the ETL pipeline that 1) initialize the DB, 2) extract the coin list information, 3) load the coin list to DB, 4) extract the historical information for the coins, 5) load this information into the DB
run_etl()

st.write("""
# Heading
This is the Cryptovaluer App. We apply some technical analysis and ML to cryptocoins to provide information regarding trends and opportunities. 

This is not financial advise!""")