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

pd.set_option('display.max_rows', None)  # None means unlimited
pd.set_option('display.max_columns', None)

with open('financial_data.pkl', 'rb') as file:
    financial_data = pickle.load(file)

with open('all_data.pkl', 'rb') as file:
    stock_prices = pickle.load(file)

with open('spxstocks.pkl', 'rb') as file:
    currentconstituents, spxchanges, spxstocks, grouped_spx = pickle.load(file)

print(financial_data['MMM'].head(10))