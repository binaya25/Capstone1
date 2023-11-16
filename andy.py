import math
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import find_dotenv, load_dotenv
import alpha_vantage
import asyncio
from rich.console import Console
from rich.markdown import Markdown


AV_API = "INSERT THE API KEY"
TICKER = 'AAPL'                           # Here I've put the symbol for Apple, please change it for your analysis
companyOverview = requests.get(f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={TICKER}&apikey={AV_API}')
companyOverview = companyOverview.json()
incomeStmt = requests.get(f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={TICKER}&apikey={AV_API}')
incomeStmt = incomeStmt.json()
balanceSheet = requests.get(f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={TICKER}&apikey={AV_API}')
balanceSheet = balanceSheet.json()
cashflow = requests.get(f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={TICKER}&apikey={AV_API}')
cashflow = cashflow.json()

print(f"Analysing {companyOverview['Name']}, {companyOverview['Address']}\n{companyOverview['Description']}")
pe_ratio = float(companyOverview['PERatio'])
peg_ratio = float(companyOverview['PEGRatio'])
pe_ratio = float(companyOverview['PERatio'])
beta = float(companyOverview['Beta'])
MA_50 = float(companyOverview['50DayMovingAverage'])
MA_200 = float(companyOverview['200DayMovingAverage'])

annual_incomes = pd.DataFrame.from_records(incomeStmt['annualReports'])
annual_incomes.set_index('fiscalDateEnding', inplace=True)
annual_incomes = annual_incomes.apply(pd.to_numeric, errors='coerce')
annual_incomes.dropna(axis=1, inplace=True)
annual_incomes

annual_balances = pd.DataFrame.from_records(balanceSheet['annualReports'])
annual_balances.set_index('fiscalDateEnding', inplace=True)
annual_balances = annual_balances.apply(pd.to_numeric, errors='coerce')
annual_balances.dropna(axis=1, inplace=True)
annual_balances

annual_cashflow = pd.DataFrame.from_records(cashflow['annualReports'])
annual_cashflow.set_index('fiscalDateEnding', inplace=True)
annual_cashflow = annual_cashflow.apply(pd.to_numeric, errors='coerce')
annual_cashflow.dropna(axis=1, inplace=True)
annual_cashflow

MARKDOWN = f"""
# P/E Ratio : {pe_ratio} \t PEG Ratio: {peg_ratio} \t Beta : {beta} \n # Signal 1: {'BUY (50 day Moving Average is above 200 day)' if MA_50 > MA_200 else 'SELL (50 day Moving Average is below 200 day)'}

"""

console = Console()
md = Markdown(MARKDOWN)
console.print(md)
#
fig1 = plt.figure(figsize=(15, 3))
plt.plot(annual_incomes.index, annual_incomes.operatingIncome, label='Annual Operating Income (Left scale)', c='g')
ax = fig1.axes[0]
ax2 = ax.twinx()
ax2.plot(annual_incomes.index, annual_incomes.operatingExpenses, label='Annual Operating Expense (Right scale)', c='r')
_ = plt.title("Income vs Expense Chart")
ax.legend(loc='lower left')
ax2.legend(loc='upper right')
ax.annotate('Both income and Expense increasing, but income scale is larger than expense', xy=('2021-09-30', 1.2e11),
           xytext=('2020-09-30', 0.9e11), arrowprops=dict(facecolor='black', shrink=0.05), bbox=dict(boxstyle='round', fc='0.8'))
ax.set_ylabel('USD')
ax2.set_ylabel('USD')
#
income_vs_expense = ((annual_incomes.operatingIncome / annual_incomes.operatingExpenses) - 1).mean()

width = 0.25
multiplier = 0
date_list = None
fig2, ax = plt.subplots(layout='constrained', figsize=(12.75, 3))
for attr in annual_balances[['totalAssets', 'totalLiabilities']].items():
    offset = width * multiplier
    if date_list is None:
        date_list = attr[1].index.tolist()
    x = np.arange(len(date_list))
    rects = ax.bar(x + offset, attr[1].values.tolist(), width, label=attr[0])
    # ax.bar_label(rects, padding=3)
    multiplier += 1
ax.set_ylabel('USD')
ax.set_title("Gross Profit Chart")
ax.set_xticks(x + width/2, date_list)
ax.legend(loc='upper right')

asset_vs_liability = ((annual_balances.totalAssets / annual_balances.totalLiabilities) - 1).mean()

fig3 = plt.figure(figsize=(15.93, 3))
rects = plt.bar(annual_cashflow.index, annual_cashflow.profitLoss, label='Profit/Loss')
ax = fig3.axes[0]
_ = ax.bar_label(rects, padding=3)

pnl = (annual_cashflow['profitLoss'] > 0).mean()

signals = np.mean([sg > 0 for sg in [pnl, asset_vs_liability, income_vs_expense]])

if signals > 0.5:
    console = Console()
    md = Markdown(f"# Signal 2: {'BUY'}")
    console.print(md)
else:
    console = Console()
    md = Markdown(f"# Signal 2: {'SELL'}")
    console.print(md)

# Fetch news sentiment using Alpha Vanatage
def FN1(topic, stock_code):
    #find each ticker score
    for ticker in topic['ticker_sentiment']:
            if (stock_code == ticker['ticker']):
                return{f"{stock_code}-ticker_sentiment_score": ticker['ticker_sentiment_score']}
    return topic

def topicBase(feed_decode, stock_code):
    #de-structure topic base on topic
    result = []
    for eachTopic in feed_decode:
        result.append(FN1(eachTopic, stock_code))
    return result

async def main(stock_code):
    api_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={TICKER}&apikey={AV_API}'
    feed = requests.get(api_url)
    feed_decode = feed.json()

    return topicBase(feed_decode['feed'], stock_code)

print(asyncio.run(main('AAPL')))

# Calculate overall sentiment score
sentiment_score = asyncio.run(main('AAPL'))
for item in sentiment_score:
    print(item.get('AAPL-ticker_sentiment_score'))


x = float(item.get('AAPL-ticker_sentiment_score'))
y = len(sentiment_score)
overall_sentiment_score = math.fsum([x])/y # surround the arguments with square brackets since SUM funciton expects a list
print(overall_sentiment_score)

#Display Signal 3
console = Console()
if overall_sentiment_score > 0.15:
    md = Markdown("# Signal 3: BUY")
elif -0.15 <= overall_sentiment_score <= 0.15:
    md = Markdown("# Signal 3: HOLD")
else:
    md = Markdown("# Signal 3: SELL")
console.print(md)