import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import pickle

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

#Test

pd.set_option('display.max_rows', None)  # None means unlimited
pd.set_option('display.max_columns', None)

def get_tickers():

    # Scrape the Wikipedia page
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the table with id 'constituents'
    table = soup.find('table', {'id': 'constituents'})

    # Convert the table to a DataFrame
    currentconstituents = pd.read_html(str(table))[0]

    # Extract the table with id 'changes'
    changes_table = soup.find('table', {'id': 'changes'})

    # Convert the table to a DataFrame
    spxchanges = pd.read_html(str(changes_table))[0]

    # Rename columns based on the provided R code and remove the first two rows
    spxchanges.columns = ['Date', 'AddTicker', 'AddName', 'RemovedTicker', 'RemovedName', 'Reason']
    spxchanges = spxchanges.iloc[2:].reset_index(drop=True)

    # Convert the 'Date' column to a datetime format
    spxchanges['Date'] = pd.to_datetime(spxchanges['Date'], format='%B %d, %Y')

    # Extract year and month columns
    spxchanges['year'] = spxchanges['Date'].dt.year
    spxchanges['month'] = spxchanges['Date'].dt.month

    # Create the month sequence
    currentmonth = pd.Timestamp(datetime.datetime.now().replace(day=1))
    monthseq = pd.date_range(start='1990-01-01', end=currentmonth, freq='MS')[::-1]

    # Initialize spxstocks DataFrame
    spxstocks = currentconstituents[['Symbol', 'Security']].copy()
    spxstocks.columns = ['Ticker', 'Name']
    spxstocks['Date'] = currentmonth
    lastrunstocks = spxstocks

    # Iterate through months, working backward in time
    for d in monthseq[1:]:
        y, m = d.year, d.month
        changes = spxchanges[(spxchanges['year'] == y) & (spxchanges['month'] == m)]

        # Remove added tickers
        tickerstokeep = lastrunstocks[~lastrunstocks['Ticker'].isin(changes['AddTicker'])].copy()
        tickerstokeep['Date'] = d

        # Add back the removed tickers
        tickerstoadd = changes[changes['RemovedTicker'].notnull()][['Date', 'RemovedTicker', 'RemovedName']]
        tickerstoadd.columns = ['Date', 'Ticker', 'Name']

        # Combine the data for this month
        thismonth = pd.concat([tickerstokeep, tickerstoadd], ignore_index=True)
        spxstocks = pd.concat([spxstocks, thismonth], ignore_index=True)

        lastrunstocks = thismonth

        grouped_spx = spxstocks.groupby('Ticker').first().reset_index()

        return currentconstituents, spxchanges, spxstocks, grouped_spx

def get_stock_price_data(ticker):
    print('Now retrieving data for ', ticker)
    stock = yf.Ticker(ticker)
    # Get historical market data, here max is used to get all the available data
    hist = stock.history(period="max")
    return hist


def get_financial_data(ticker, driver):

    print('Now attempting to retrieve data for', ticker)

    url = f"https://stockanalysis.com/stocks/{ticker.lower()}/financials/?p=quarterly"
    driver.get(url)

    # Wait for the table to load
    try:
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table[data-test='financials']"))
        )
    except:
        print('Unable to retrieve data... (probably ticker is inactive)')
        return None

    # Rest of the code remains the same...

    headers = driver.find_elements(By.CSS_SELECTOR, "table[data-test='financials'] thead th")
    columns = [header.text for header in headers]

    rows = driver.find_elements(By.CSS_SELECTOR, "table[data-test='financials'] tbody tr")
    data = []
    for row in rows:
        data.append([cell.text for cell in row.find_elements(By.TAG_NAME, "td")])

    df = pd.DataFrame(data, columns=columns)
    df = df.transpose()

    return df


currentconstituents, spxchanges, spxstocks, grouped_spx = get_tickers()

with open('spxstocks.pkl', 'wb') as file:
    pickle.dump([currentconstituents, spxchanges, spxstocks, grouped_spx], file)




# Let's assume spxstocks['Ticker'] contains the tickers of the S&P 500 companies
stock_prices = {}
for ticker in grouped_spx['Ticker']:
    stock_prices[ticker] = get_stock_price_data(ticker)

# Save the all_data dictionary to a pickle file
with open('stock_prices.pkl', 'wb') as file:
    pickle.dump(stock_prices, file)



# Path to the ChromeDriver executable
CHROMEDRIVER_PATH = r"/usr/local/bin/chromedriver/chromedriver"

chrome_options = Options()
# Mimic a real browser request
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH, options=chrome_options)

# Dictionary to store data
financial_data = {}

for ticker in grouped_spx['Ticker']:
    financial_data[ticker] = get_financial_data(ticker, driver)

driver.close()

with open('financial_data.pkl', 'wb') as file:
    pickle.dump(financial_data, file)


# print(currentconstituents.head(5))
# print(spxchanges.head(5))
# print(spxstocks.head(5))
# print(all_data['MMM'].head(5))