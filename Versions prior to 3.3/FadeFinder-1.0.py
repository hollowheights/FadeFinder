#---------------------------------- IMPORTS AND SIMILAR ----------------------------------
import datetime
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
timemeasure = datetime.datetime.now()
pd.set_option('display.max_columns', None)
from scipy.stats.mstats import gmean


#-------------- Set up test universe - loaded from local SQLite database ----------------------

#connect to SQLite database
connect = sqlite3.connect("SRUSListedVersion2.db")

#create a cursor
c = connect.cursor()

#add all symbols in the database to 'stocklist'
stocklist = []

c.execute('''select * FROM sqlite_master WHERE type="table"''')
storagelist = c.fetchall()

for x in range(len(storagelist)):
    stocklist.append(storagelist[x][1])

stocklist = stocklist[0:600] #Limit test to first x stocks in test universe


#-------------------- Parameters for the backtest---------------------
startdate = "2016-01-01"
enddate = "2021-09-09"
direction = "short" #"short"/"long"
Exposure = 0.1 #percentage of portfolio to use for each trade

# 0=off, 1=on
PriceChangeFilter = 1 #Change from open to close relative to average daily range
PriceChangeSetting = 8

RVOLFilter = 0 #Relative volume
RVOLSetting = 2

DollarVolFilter = 1 #Absolute volume to secure liquidity
DollarVolSetting = 10000

PercentileFilter = 0 #Close in percentile of day's range
PercentileSetting = 50

SharePriceFilter = 1 #Exclude the cheapest stocks - possibly redundant
SharePriceSetting = 1



#------------------------ Actual testing section ----------------------------------

#Lists for data storage - to build 'resultsdataframe' from
list_dates = []
list_stocknames = []
list_changes = []
list_changesrel = []
list_dollarvolumes = []
list_day1 = []
list_day2 = []
list_overnight = []
list_day1Trades = []
list_day2Trades = []
list_overnightTrades = []
list_RVOL10D = []

#List for storing symbols of stocks with data errors
list_failedstocks = []


#Definition of Tradefunction
def TradeFunction(x,stock):
    Tradesignal = 1

    change = df.loc[x, "close"] / df.loc[x, "open"]
    changerel = (df.loc[x, "close"] - df.loc[x, "open"]) / df["range"].mean()

    if PriceChangeFilter == 1:
        if changerel < PriceChangeSetting:
            Tradesignal = 0

    if RVOLFilter == 1:
       if df.loc[x,"volume"] < RVOLSetting*df.loc[x,"avgvolume10D"]:
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
            list_dates.append(df["date"][x])
            list_stocknames.append(stock)
            list_changes.append(change)
            list_changesrel.append(changerel)
            list_RVOL10D.append(df.loc[x,"volume"] / df.loc[x,"avgvolume10D"])#df.loc[x,"volume"]/df.loc[x, "avgvolume10D"])
            list_dollarvolumes.append(df["$volume"][x])

        except:
            print("Failed to log one of: date,entryprice,stockname,pricechange,RVOL")

        if direction == "long":
            try:
                list_day1.append(df.loc[x + 1, "close"] / df.loc[x, "close"]-1)
                list_day2.append(df.loc[x + 2, "close"] / df.loc[x + 1, "close"]-1)
                list_overnight.append(df.loc[x + 1, "open"] / df.loc[x, "close"]-1)

                list_day1Trades.append(((df.loc[x + 1, "close"] / df.loc[x, "close"])-1 * Exposure))
                list_day2Trades.append(((df.loc[x + 2, "close"] / df.loc[x+1, "close"])-1 * Exposure))
                list_overnightTrades.append(((df.loc[x + 1, "open"] / df.loc[x, "close"])-1 * Exposure))

            except:
                print("Failed to log results for a long trade in stock: %s" %stock)

        elif direction == "short":

            try:
                list_day1.append(((df.loc[x,"close"]-df.loc[x+1,"close"])/ df.loc[x, "close"]))
                list_day2.append(((df.loc[x+1,"close"]-df.loc[x+2,"close"])/ df.loc[x+1, "close"]))
                list_overnight.append(((df.loc[x, "close"] - df.loc[x + 1, "open"])/df.loc[x, "close"]))

                list_day1Trades.append((((df.loc[x, "close"] - df.loc[x + 1, "close"]) / df.loc[x, "close"]) * Exposure) )
                list_day2Trades.append((((df.loc[x + 1, "close"] - df.loc[x + 2, "close"]) / df.loc[x +1 , "close"]) * Exposure))
                list_overnightTrades.append((((df.loc[x, "close"] - df.loc[x + 1, "open"])/df.loc[x, "close"]) * Exposure))

            except:
                print("Failed to log results for a short trade in stock: %s" % stock)

        else:
            print("Direction for trade needs a parameter setting")



#------------------------- Initiate the backtest ------------------------
for stock in stocklist:
    try:
        # Load data from SQLite and set up the dataframe
        df = (pd.read_sql("SELECT * FROM %s WHERE DATE(date) BETWEEN '%s' AND '%s' " %(stock,startdate,enddate), connect))
        pd.set_option("display.max_rows", 300, "display.min_rows", 200, "display.max_columns", None, "display.width", None)

        #Run function with itertuples
        for row in df.iloc[:-2].itertuples():
            TradeFunction(row.Index, stock)


    except:
        print("Error with stock: %s" % stock)
        list_failedstocks.append(stock)



#-----------------------------------Set up dataframe for results --------------------------------------

resultsdataframe = pd.DataFrame({"Date": list_dates,
                                 "Stock": list_stocknames,
                                 "Change": list_changes,
                                 "Changerel": list_changesrel,
                                 "RVOL10D": list_RVOL10D,
                                 "$volume": list_dollarvolumes,
                                 "Day1": list_day1,
                                 "Day2": list_day2,
                                 "Overnight": list_overnight,
                                 "Day1Trade": list_day1Trades,
                                 "Day2Trade": list_day2Trades,
                                 "OvernightTrade": list_overnightTrades})

resultsdataframe["Day1TradeCum"] = resultsdataframe["Day1Trade"].cumsum()
resultsdataframe["Day2TradeCum"] = resultsdataframe["Day2Trade"].cumsum()
resultsdataframe["OvernightTradeCum"] = resultsdataframe["OvernightTrade"].cumsum()


print("resultsdataframe:")
print(resultsdataframe.head(20)) #Print first x rows of resultsdataframe for visual inspection



#---------------------------------------------------------------------------------------------------------------
#                  At this point all data should be the same for version 1 and version 2                       #
#---------------------------------------------------------------------------------------------------------------



#--------------------- Preparing data for presentation -----------------------------------------------


try:
    Logdataframe = pd.DataFrame({
        "Test date": [datetime.date.today()],
        "Direction": [direction],
        "# Stocks": [len(stocklist)],
        "# Days": [np.busday_count(startdate, enddate)],
        "Total rows": [np.busday_count(startdate, enddate) * len(stocklist)],
        "Trade signals": [resultsdataframe.shape[0]],
        "Signal frequency": ["1 / %s" % ((np.busday_count(startdate, enddate) * len(stocklist)) / resultsdataframe.shape[0])],
        "RVOL setting": [RVOLSetting],
        "Price change setting": [PriceChangeSetting],
        "Range percentile setting": [PercentileSetting] })

except:
    "Error setting up 'Logdataframe'"



#--------------------------------- Results analysis ------------------------------

#Print first 20 lines of resultsdataframe
print("resultsdataframe: \n",resultsdataframe.head(20))

#Print the dataframe holding parameter settings and results
print("\n",Logdataframe.to_string(index=False))


#Stats on stocks failing
print("Failure list's length now: ", len(list_failedstocks))

#Runtime of script - placed before plots due to blocking tendencies
print("Time to run script: ", datetime.datetime.now()-timemeasure)



#-------------------------------------- Plotting charts --------------------------------------------------

resultsdataframe["Day1TradeCum"].plot(legend="Day1TradeCum")
resultsdataframe["Day2TradeCum"].plot(legend="day2TradeCum",color="black")
resultsdataframe["OvernightTradeCum"].plot(legend="OvernightTradeCum")
plt.show()

resultsdataframe.plot(x="RVOL10D",y="Day1TradeCum", legend= "trade return", style="o")
plt.show()

#resultsdataframe.Day1Trade.plot.hist(by=resultsdataframe.RVOL10D,bins=50,grid=True, xlabel="Testio")




#------------------------------ Logging to files  -------------------------------------------------

try:
    #Testresults added to previous results
    #pd.DataFrame.to_csv(Logdataframe,r"C:\Users\LENOVO\desktop\testfilmarts2.csv",mode="a",header=True,index=False)

    #Store latest instance of resultsdataframe temporary - useful when comparing versions
    with pd.ExcelWriter("resultsdataframe.xlsx") as writer:
        resultsdataframe.to_excel(writer)

    #Store list of symbols for stocks with data errors
    FailureListStorage = open('FailureList.txt','w')
    FailureListStorage.write(str(list_failedstocks))
    FailureListStorage.close()

except:
    print("Error when logging results to spreadsheet")



#------------------------------- Scrap code --------------------------------------------------------


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

