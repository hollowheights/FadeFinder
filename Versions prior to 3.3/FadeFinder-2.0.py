#IMPORTS AND SIMILAR
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

stocklist = stocklist[0:200] #Limit test to first x stocks in test universe


#-------------------- Parameters for the backtest---------------------
startdate = "2016-01-01"
enddate = "2021-09-09"
direction = "short" #"short"/"long"
Exposure = 0.1 #percentage of portfolio to use for each trade

# 0=off, 1=on
PriceChangeFilter = 1 #Change from open to close relative to average daily range
PriceChangeSetting = 5

RVOLfilter = 0 #Relative volume
RVOLSetting = 2

DollarVolfilter = 1 #Absolute volume to secure liquidity
DollarVolSetting = 10000

PercentileFilter = 0 #Close in percentile of day's range
PercentileSetting = 50

SharePriceFilter = 1 #Exclude the cheapest stocks - possibly redundant
SharePriceSetting = 1



#------------------- Actual testing section----------------------------------

#store list to build resultsdataframe from
resultspartialdataframes = []

#List for storing symbols of stocks with data errors
list_failedstocks = []

#Definition of Tradefunction
def TradeFunction(stock):

    #Define checks and flag all rows giving a trade signal

    PriceChangeParam = ((df.close-df.open)/df.range.mean()) > PriceChangeSetting
    DollarVolParam = df["$volume"] > DollarVolSetting
    SharePriceParam = df.close > SharePriceSetting

    TradeSignalParam = (PriceChangeParam & DollarVolParam & SharePriceParam) == True
    df["TradeSignal"] = np.where(TradeSignalParam,1,0)
    df["ChangeRel"] = np.where(TradeSignalParam,(df.close-df.open)/df.range.mean(),0)
    df["RVOL10D"] = np.where(TradeSignalParam,df.volume/df.avgvolume10D,0)

    #Calculate returns for these dates and for simulated trades

    if direction == "long":
        df["Day1"] = np.where(TradeSignalParam > 0,(df.close.shift(-1)/df.close) -1 ,0) #return close to close (including overnight)
        df["Day2"] = np.where(TradeSignalParam > 0, (df.close.shift(-2) / df.close.shift(-1))-1, 0) #close to close day T+1 to T+2
        df["Overnight"] = np.where(TradeSignalParam > 0, (df.open.shift(-1)/df.close)-1 ,0) # return close to close (including overnight)


        df["Day1Trade"] = np.where(TradeSignalParam > 0, (((df.close.shift(-1)/df.close)-1) * Exposure),0)  # close to close day T+1 to T+2
        df["Day2Trade"] = np.where(TradeSignalParam > 0, (((df.close.shift(-2)/df.close.shift(-1))-1) * Exposure),0)  # close to close day T+1 to T+2
        df["OvernightTrade"] = np.where(TradeSignalParam > 0, (((df.open.shift(-1)/df.close)-1) * Exposure),0)  # close to close day T+1 to T+2


    elif direction == "short":
        df["Day1"] = np.where(TradeSignalParam > 0, (df.close - df.close.shift(-1)) / df.close,0)
        df["Day2"] = np.where(TradeSignalParam > 0, (df.close.shift(-1) - df.close.shift(-2)) / df.close.shift(-1),0)
        df["Overnight"] = np.where(TradeSignalParam > 0, (df.close-df.open.shift(-1))/df.close,0)

        df["Day1Trade"] = np.where(TradeSignalParam > 0, df.Day1 * Exposure,0)
        df["Day2Trade"] = np.where(TradeSignalParam > 0, df.Day2 * Exposure,0)
        df["OvernightTrade"] = np.where(TradeSignalParam > 0, df.Overnight * Exposure,0)


    resultspartialdataframes.append(df.loc[df.TradeSignal == 1])



#------------------------- Initiate the backtest ------------------------
for stock in stocklist:
    try:
        # Load data from SQLite and set up the dataframe
        df = (pd.read_sql("SELECT * FROM %s WHERE DATE(date) BETWEEN '%s' AND '%s' " %(stock,startdate,enddate), connect))
        df.drop(["average","barCount"],axis=1, inplace=True)
        df.insert(0,"Symbol",stock)
        pd.set_option("display.max_rows", 300, "display.min_rows", 200, "display.max_columns", None, "display.width", None)


        TradeFunction(stock)


    except:
        print("Error with stock: %s" % stock)
        list_failedstocks.append(stock)



#---------------------Set up  dataframe for results----------------------------

#create dataframe
resultsdataframe = pd.concat(resultspartialdataframes, ignore_index=True)

#Add column for "Change"
resultsdataframe.insert(3,"Change", resultsdataframe.close/resultsdataframe.open)

#Delete unnecessary columns
resultsdataframe.drop(["high", "open", "low","close","range","TradeSignal","volume","avgvolume30D","avgvolume10D","percentileclose"], axis=1, inplace=True)
resultsdataframe.rename({"Symbol":"Stock","date":"Date"}, axis=1, inplace=True)

#Add 3 columns with cumulative results
resultsdataframe["Day1TradeCum"] = resultsdataframe["Day1Trade"].cumsum()
resultsdataframe["Day2TradeCum"] = resultsdataframe["Day2Trade"].cumsum()
resultsdataframe["OvernightTradeCum"] = resultsdataframe["OvernightTrade"].cumsum()

#Set order of dataframe columns
indexlist = ["Date","Stock","Change","ChangeRel","RVOL10D","$volume","Day1","Day2","Overnight","Day1Trade","Day2Trade","OvernightTrade","Day1TradeCum","Day2TradeCum","OvernightTradeCum"]
resultsdataframe = resultsdataframe.reindex(indexlist, axis=1)



#---------------------------------------------------------------------------------------------------------------
#                  At this point all data should be the same for version 1 and version 2                       #
#---------------------------------------------------------------------------------------------------------------



#--------------------- Preparing data for presentation ----------------------------------------------------------------

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



#----------------------------------- Plotting -------------------------------

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




#------------------------- Not in use ------------------------

'''

    #df["PriceChangeFlag"] = np.where(PriceChangeParam,1,0)
    #df["DollarVolFlag"] = np.where(DollarVolParam,1,0)
    #df["SharePriceFlag"] = np.where(SharePriceParam,1,0)


'''
