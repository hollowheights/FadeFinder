#---------------------------------- IMPORTS AND SIMILAR ----------------------------------
import datetime
import sqlite3
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean
import pandas as pd
import numpy as np
timemeasure = datetime.datetime.now()
pd.set_option('display.max_columns', None)



#-------------- Set up test universe - loaded from local SQLite database ----------------------

#connect to SQLite database
connect = sqlite3.connect("DatabaseName")

#create a cursor
c = connect.cursor()

#add all symbols in the database to 'stocklist'
stocklist = [] #stockbuffer

c.execute('''select * FROM sqlite_master WHERE type="table"''')
storagelist = c.fetchall()

for x in range(len(storagelist)):
    stocklist.append(storagelist[x][1])

stocklist = stocklist[0:300] #Limit test to first x stocks in test universe



#-------------------- Parameters for the backtest---------------------
startdate = "'2016-01-01'"
enddate = "'2021-09-09'"

# 0=off, 1=on
PriceChangeFilter = 1 #Change from open to close relative to average daily range
PriceChangeSetting = 5

RVOLFilter = 0 #Relative volume
RVOLSetting = 2

DollarVolFilter = 1 #Absolute volume to secure liquidity
DollarVolSetting = 10000

PercentileFilter = 0 #Close in percentile of day's range
PercentileSetting = 50

SharePriceFilter = 1 #Exclude the cheapest stocks - possibly redundant
SharePriceSetting = 1



#------------------------ Actual testing section ----------------------------------

#Lists for data storage - to build 'results dataframe' from
dates = []
stocknames = []
entryprices = []
pricechangespct = []
pricechangesrelative = []

overnightreturns = []
day1returns = []
day2returns = []
RVOL10D = []
FailureList = []

#Definition of Tradefunction
def TradeFunction(x,stock):
    Tradesignal = 1

    if PriceChangeFilter == 1:
        pricechangepct = df.loc[x,"close"]/df.loc[x,"open"]
        pricechangerelative = (df.loc[x,"close"] - df.loc[x,"open"]) / df["range"].mean()
        if pricechangerelative < PriceChangeSetting:
            Tradesignal = 0

    if RVOLFilter == 1:
       if df.loc[x,"volume"] > RVOLSetting*df.loc[x,"avgvolume10D"]:
            Tradesignal = 0

    if DollarVolFilter == 1:
        if df.loc[x, "$volume"] < DollarVolSetting:
            Tradesignal = 0

    if PercentileFilter == 1:
        if df.loc[x,"percentileclose"] < PercentileSetting:
            Tradesignal = 0

    if SharePriceFilter == 1:
        if df.loc[x,"close"] < SharePriceSetting:
            Tradesignal = 0


    #log trade
    if Tradesignal == 1:

        try:
            dates.append(df["date"][x])
            entryprices.append(df["close"][x])
            stocknames.append(stock)
            pricechangespct.append(pricechangepct)
            pricechangesrelative.append(pricechangerelative)
            RVOL10D.append(df.loc[x,"volume"] / df.loc[x,"avgvolume10D"])#df.loc[x,"volume"]/df.loc[x, "avgvolume10D"])

        except:
            print("Failed to log one of: date,entryprice,stockname,pricechange,RVOL")

        try:
            day1returns.append(df.loc[x + 1, "close"] / df.loc[x, "close"])                day2returns.append(df.loc[x + 2, "close"] / df.loc[x + 1, "close"])
            overnightreturns.append(df.loc[x + 1, "open"] / df.loc[x, "close"])

            day1returns.append(((df.loc[x,"close"]-df.loc[x+1,"close"])/ df.loc[x, "close"])+1)
            day2returns.append(((df.loc[x+1,"close"]-df.loc[x+2,"close"])/ df.loc[x+1, "close"])+1)
            overnightreturns.append(df.loc[x, "close"] / df.loc[x + 1, "open"])
        except:
            print("Failed to log results for a trade in stock: %s" %stock)

        else:
            print("Direction for trade needs a parameter setting")



#------------------------- Initiate the backtest ------------------------
for stock in stocklist:
    try:
        # Load data from SQLite and set up the dataframe
        df = (pd.read_sql("SELECT * FROM %s WHERE DATE(date) BETWEEN %s AND %s" %(stock,startdate,enddate), connect))
        pd.set_option("display.max_rows", 300, "display.min_rows", 200, "display.max_columns", None, "display.width", None)

        #Run function with itertuples
        for row in df.iloc[:-2].itertuples():
            TradeFunction(row.Index, stock)


    except:
        print("Error with stock: %s" % stock)
        FailureList.append(stock)



#-----------------------------------Set up dataframe for results --------------------------------------

resultsDataFrame = pd.DataFrame({"Date": dates,
                                 "Stockname": stocknames,
                                 "Pricechangepct": pricechangespct,
                                 "Pricechangerelative": pricechangesrelative,
                                 "RVOL10D": RVOL10D,
                                 "Entry price": entryprices,
                                 "day1return": day1returns,
                                 "day2return": day2returns,
                                "overnightreturn": overnightreturns})

resultsDataFrame["day1return10%RiskCum"] = (((resultsDataFrame["day1return"]-1)/10)+1).cumprod()
resultsDataFrame["day2return10%RiskCum"] = (((resultsDataFrame["day2return"]-1)/10)+1).cumprod()
resultsDataFrame["overnightreturn10%RiskCum"] = (((resultsDataFrame["overnightreturn"]-1)/10)+1).cumprod()


print("resultsDataFrame:")
print(resultsDataFrame.head(20)) #Print first x rows of resultsdataframe for visual inspection


#--------------------- Preparing data for presentation ----------------------------------------------------------------
#Variables for analysis of results
try:
    NumberTradingDays = len(df) #wrong if last stock tested doesn't have date for entire backtest period
    NumberTradingDaysXTickers = len(stocklist*NumberTradingDays)
    NumberTradeSignals = len(day1returns)
    SignalFrequency = "1 / %s"%(NumberTradingDaysXTickers/NumberTradeSignals)

    GeoMeanOvernight = gmean(overnightreturns)
    GeoMeanDay1 = gmean(day1returns)
    GeoMeanDay2 = gmean(day2returns)
except:
    print("Error defining variables with results")

try:
    Logdataframe1 = pd.DataFrame({
        "Direction tested": [direction],
        "Number of stocks tested": [len(stocklist)],
        "Number of trading days": [NumberTradingDays],
        "Trading days X tickers": [NumberTradingDaysXTickers],
        "Trading signals": [NumberTradeSignals],
        "Signal frequency": [SignalFrequency]})
except:
    print("Error setting up Logdataframe1")


try:
    Logdataframe2 = pd.DataFrame({
        "Date": [datetime.date.today()],
        "RVOL setting": [RVOLSetting],
        "Price change setting": [PriceChangeSetting],
        "Range percentile setting": [PercentileSetting],
        "Geo mean overnight": [GeoMeanOvernight],
        "Geo mean day 1": [GeoMeanDay1],
        "Geo mean day 2": [GeoMeanDay2]},
        columns = ["Date", "RVOL setting", "Price change setting", "Range percentile setting",
        "Geo mean overnight", "Geo mean day 1", "Geo mean day 2"])

except:
    "Error setting up 'Logdataframe2'"



#--------------------- Presentation of data ----------------------------------------------------------------------------

#Print the two dataframes holding parameter settings and results
try:
    print("\n",Logdataframe1.to_string(index=False))
    print("\n",Logdataframe2.to_string(index=False),"\n")
except:
    print("Error when presenting results")

#Stats on stocks failing and storing a list of failed symbols
print("Failure list's length now: ", len(FailureList))

FailureListStorage = open('FailureList.txt','w')
FailureListStorage.write(str(FailureList))
FailureListStorage.close()

#Runtime of script - placed before plots due to blocking tendencies
print("Time to run script: ",datetime.datetime.now()-timemeasure)



#-------------------------------------- Plotting charts --------------------------------------------------

resultsDataFrame["day2return10%RiskCum"].plot(legend="day2returnRiskControlCumulative")
resultsDataFrame["day1return10%RiskCum"].plot(legend="day1return",color="black")
resultsDataFrame["day2return10%RiskCum"].plot(legend="day2return")

plt.show()

resultsDataFrame.plot(x="RVOL10D",y="day1return", legend= "trade return", style="o")
plt.show()



#--------------------- Logging results to CSV ---------------------------------------------------------------------------
try:
    pd.DataFrame.to_csv(Logdataframe,r"C:\Users\LENOVO\desktop\testfilmarts2.csv",mode="a",header=True,index=False)
    #Logdataframe.to_excel(r"C:\Users\LENOVO\desktop\11martsBacktesting2.xlsx",index=False, header=False,mode="a")

except:
    print("Error when logging results to .csv")



#--------------------------- Scrap code --------------------------------------------------------
'''
        

#-------------- Check data for resultsdataframe ---------------------

#useful when data goes out of index and setup of dataframe throws an error

#print("Print of all rows with nan values: \n")
#print(df[df.isna().any(axis=1)])
#print("Length before dropna:",len(df))
#df = df.dropna()  # Remove all rows in df with NA/NAN values
#print("length after dropna:",len(df))


print("length of dates:", len(dates))
print("length of stockname:", len(stocknames))
print("length of entry price:", len(entryprices))
print("length of overnightreturn:", len(overnightreturns))
print("length of day1:", len(day1returns))
print("length of day2:", len(day2returns))


'''

