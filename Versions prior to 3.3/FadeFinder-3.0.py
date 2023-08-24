# --- IMPORTS AND SIMILAR ---#
import datetime
from dateutil import tz
import sqlite3
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats.mstats import gmean
pd.set_option("display.max_rows", 1000, "display.min_rows", 200, "display.max_columns", None, "display.width", None)
pd.options.display.width = 0

timemeasure = time.perf_counter()


def timezone_convert(time):
    return time.replace(tzinfo=tz.UTC).astimezone(tz.gettz('America/New_York'))


# -------------- Set up test universe - loaded from local SQLite db --------------------
#connect to SQLite database
connect = sqlite3.connect(r"C:\01 SR Data partition\TradingDatabaser\SRUSListedPolygonDaily.db")
c = connect.cursor()

# add all symbols in the database to 'stocklist'
stocklist = []
c.execute('''select * FROM sqlite_master WHERE type="table"''')
storagelist = c.fetchall()

for x in range(len(storagelist)):
    stocklist.append(storagelist[x][1])

# stocklist = stocklist[0:100] #Limit test to first x stocks in test universe


# -------------------- Parameters for the backtest---------------------
startdate = "2018-01-01"
enddate = "2021-12-31"
Signalperiod = "gap"  #"gap" / "day"
Exposure = 0.1  #percentage of portfolio to use for each trade
Fees = 0.005  #Percentage - 1=100%
direction = "short"  # "short" / "long"

# 0=off, 1=on
PriceChangeFilter = 1  # Change from open to close in percentage
PriceChangeSetting = 15

RVOLfilter = 0  # Relative volume
RVOLSetting = 2

DollarVolfilter = 1  # Absolute volume to secure liquidity
DollarVolSetting = 10_000

PercentileFilter = 0  # Close in percentile of day's range
PercentileSetting = 50

SharePriceFilter = 1  # Exclude the cheapest stocks - possibly redundant
SharePriceSetting = 1

MarketCapFilter = 0
MarketCapSetting = 500_000_000


# ------------------- Actual testing section----------------------------------

resultspartialdataframes = []  # store list to build resultsdataframe from
list_failedstocks = []  # List for storing symbols of stocks with data errors

# columns in database:
'''['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'marketcap', 'outstanding']'''

def TradeFunctionDaily(stock):
    # Create dataframe
    df = (pd.read_sql("SELECT * FROM %s WHERE DATE(date) BETWEEN '%s' AND '%s' " % (stock, startdate, enddate), connect, parse_dates=['date']) )

    # Add columns to df
    df['$volume'] = df['volume'] * abs(df['close'] - df['open'])
    df['range'] = abs((df['close'] - df['open'])) / df['open']
    df['percentileclose'] = 1 - ((df['high']-df['close']) / df['range'])
    df['avgvolume10D'] = df['volume'].rolling(10).mean()

    # Define booleans for checks that are invariant between long/short test
    DollarVolParam = df.volume.shift(1) > DollarVolSetting
    SharePriceParam = df.open > SharePriceSetting
    MarketCapParam = (df.market_cap.shift(1) < MarketCapSetting) #& (df.market_cap < MarketCapSetting)

    #Define PriceChangeParam depending on "signalperiod"
    if Signalperiod == "day":
        PriceChangeParam = ((df.close - df.open) / df.range.mean()) > PriceChangeSetting
    elif Signalperiod == "gap":
        PriceChangeParam = (((df.open - df.close.shift(1)) / df.close.shift(1)) * 100 ) > PriceChangeSetting
        DaysBreakParam = np.where(PriceChangeParam,(df.date-df.date.shift(1)) < datetime.timedelta(days=5),False)

    #Define combined filter condition for tradesignal and flag all days with a signal
    TradeSignalParam = (PriceChangeParam & DollarVolParam & SharePriceParam & DaysBreakParam) == True
    df["TradeSignal"] = np.where(TradeSignalParam,1,0)

    # Calculate returns for these dates and for simulated trades
    #shift: positive tal for at gå tilbage i tid, neg. tal for at gå frem i tid (order ascending)
    if Signalperiod == "gap":
        df["Day1"] = np.where(TradeSignalParam, (df.open - df.close) / df.open, 0)
        df["Overnight"] = np.where(TradeSignalParam, (df.close - df.open.shift(-1)) / df.open, 0)
        df["Day2"] = np.where(TradeSignalParam, (df.open.shift(-1) - df.close.shift(-1)) / df.open, 0)
        df["2Day"] = np.where(TradeSignalParam, ((df.open - df.close.shift(-1)) / df.open), 0) #day + night + day

        df["GapSize"] = np.where(TradeSignalParam, ((df.open / df.close.shift(1)) - 1)*100, 0)
        df["GapSizeRel"] = np.where(TradeSignalParam, df.GapSize / df.range.mean(), 0) #tilføj evt vægtet gns af range
        df["RVOL10D"] = np.where(TradeSignalParam,df.volume/df.avgvolume10D,0)

        df["MaxChange"] = np.where(TradeSignalParam,( (df.high / df.open) - 1) * 100, 0)
        df["OpenPrice"] = np.where(TradeSignalParam, df.open,0)
        #df["PreVolume"] = df.apply()
        #np.where(TradeSignalParam, TradeFunctionIntraday(stock,df.date),0)
        #df["PreHigh"] = np.where(TradeSignalParam, TradeFunctionIntraday(stock,df.date),0)

        #df["Weekday"] = np.where(TradeSignalParam > 0, df.date,0)
        #df["Day1"] = np.where(df.MaxChange > 0.7, -0.7, (df.open - df.close) / df.open) # -- Stop LOSS function

    # Set values for trades - based on the above
    df["Day1Trade"] = np.where(TradeSignalParam, (df["Day1"]-Fees) * Exposure,0)  # close to close day T+1 to T+2
    df["Day2Trade"] = np.where(TradeSignalParam, (df["Day2"]-Fees) * Exposure,0)  # close to close day T+1 to T+2
    df["OvernightTrade"] = np.where(TradeSignalParam, (df["Overnight"] -Fees)* Exposure,0)  # close to close day T+1 to T+2
    df["2DayTrade"] = np.where(TradeSignalParam, (df["2Day"] - Fees) * Exposure, 0)

    # store results for the stock
    resultspartialdataframes.append(df.loc[df.TradeSignal == 1])

def TradeFunctionPreVolume(stock,date):
    # set up df from intraday data + adjust time
    df = pd.read_feather(r"C:\01 SR Data partition\TradingDatabaser\SRFeather\SRUSListedPolygonIntraday\%s.feather" %stock)
    # print(df.head(10))
    # print(type(date))

    # establish a df for 3 days to reduce processing for timezone change
    day_before = datetime.datetime.combine (date - datetime.timedelta(1), datetime.datetime.min.time())
    day_after = datetime.datetime.combine (date + datetime.timedelta(1), datetime.datetime.max.time())
    df_3day = df[(df['time'] >= day_before) & (df['time'] <= day_after)]
    df_3day['time'] = df_3day['time'].map(lambda x: timezone_convert(x))

    starttime = str(date) + " 04:00"
    endtime = str(date) + " 09:30"

    CombinedCondition = (df_3day['time'] >= starttime) & (df_3day['time'] < endtime)

    # return (df[CombinedCondition]["high"].max())

    # print("Pre market high: ", df_3day[CombinedCondition]["high"].max())
    # print("Pre market total volume: ", df_3day[CombinedCondition]['volume'].sum() )
    return (df_3day[CombinedCondition]['volume'].sum())



# ------------------------- Initiate the backtest ------------------------
for stock in stocklist[200:250]:
    try:
        TradeFunctionDaily(stock)

    except:
        print("Error with stock: %s" % stock)
        list_failedstocks.append(stock)


# ---------------------Set up  dataframe for results----------------------------

#create 'resultsdataframe'
if len(resultspartialdataframes) > 1:
    resultsdataframe = pd.concat(resultspartialdataframes, ignore_index=True)
else:
    resultsdataframe = resultspartialdataframes[0]

#Delete unnecessary columns
resultsdataframe.drop(["high", "open", "low","close","range","TradeSignal","volume",
                       "avgvolume10D","percentileclose"], axis=1, inplace=True)
resultsdataframe.rename({"ticker":"Stock","date":"Date"}, axis=1, inplace=True)

#Sort resultsdataframe by date - to reflect correlation between stocks (equity curve)
resultsdataframe.sort_values(by=["Date"],ascending=True,inplace=True)
resultsdataframe.reset_index(inplace=True,drop=True)

print(resultsdataframe.head(10))


# Add premarket data: high and total volume
PreStorageList = []
for row in resultsdataframe.index:
    PreTotalVol = TradeFunctionPreVolume(resultsdataframe.loc[row, "Stock"], resultsdataframe.loc[row, "Date"])
    PreStorageList.append(PreTotalVol)
    #resultsdataframe['PreHigh'] = PreHigh
    #resultsdataframe.loc['PreVolume'] = PreTotalVol
resultsdataframe['PreVolume'] = PreStorageList

# Add 4 columns with cumulative results
resultsdataframe["Day1TradeCum"] = resultsdataframe["Day1Trade"].cumsum()
resultsdataframe["Day2TradeCum"] = resultsdataframe["Day2Trade"].cumsum()
resultsdataframe["OvernightTradeCum"] = resultsdataframe["OvernightTrade"].cumsum()
resultsdataframe["2DayTradeCum"] = resultsdataframe["2DayTrade"].cumsum()

#Set order of resultsdataframe columns
indexlist = ["Date","Stock","GapSize","GapSizeRel","RVOL10D","$volume","Day1","Day2","Overnight","2Day","Day1Trade",
             "Day2Trade","OvernightTrade","2DayTrade","Day1TradeCum","Day2TradeCum","OvernightTradeCum","2DayTradeCum", "MaxChange","OpenPrice","market_cap","PreVolume", "PreHigh"]
resultsdataframe = resultsdataframe.reindex(indexlist, axis=1)

#resultsdataframe.loc['Column_Total'] = resultsdataframe.mean(numeric_only=True, axis=0)




# ---------------------------------------------------------------------------------------------------------------
#                  At this point all data should be the same for version 1 and version 2                       #
# ---------------------------------------------------------------------------------------------------------------



#--------------------- Preparing data for presentation ----------------------------------------------------------------

try:
    Logdataframe = pd.DataFrame({
        "Test date": [datetime.date.today()],
        "Direction": [direction],
        "# Stocks": [len(stocklist)],
        "# Days": [np.busday_count(startdate, enddate)],
        "Total rows": [np.busday_count(startdate, enddate) * len(stocklist)],
        "Trade signals": [resultsdataframe.shape[0]],
        "Signal frequency": ["1 / %s" % ((np.busday_count(startdate, enddate) * len(stocklist)) / resultsdataframe.shape[0])], # OBS brug ikke stocklist som counter
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
print("Time to run script: ", time.perf_counter()-timemeasure)


#----------------------------------- Plotting -------------------------------

#Plot 3 cumulative measures
resultsdataframe["Day1TradeCum"].plot(legend="Day1TradeCum")
resultsdataframe["Day2TradeCum"].plot(legend="day2TradeCum",color="red")
resultsdataframe["OvernightTradeCum"].plot(legend="OvernightTradeCum",color="grey")
resultsdataframe["2DayTradeCum"].plot(legend = "2DayTradeCum",color="black")
plt.show()




resultsdataframe.plot(x='GapSize',y="Day1", legend= "trade return", style="o")
plt.axhline(y=0)
plt.show()

#resultsdataframe.Day1Trade.plot.hist(by=resultsdataframe.RVOL10D,bins=50,grid=True, xlabel="Testio")


#resultsdataframe[resultsdataframe.Day1Trade > 0].count()
#df = resultsdataframe[(resultsdataframe["ChangeRel"] > 10) & (resultsdataframe["Change"]<3)]

connect.close()

#------------------------------ Logging to files  -------------------------------------------------

'''
try:
    #Testresults added to previous results
    #pd.DataFrame.to_csv(Logdataframe,r"C:\ Users\LENOVO\desktop\ testfilmarts2.csv",mode="a",header=True,index=False) #OBS fjern mellemrum i path!

    #Store latest instance of resultsdataframe temporary - useful when comparing versions
    with pd.ExcelWriter("resultsdataframe.xlsx") as writer:
        resultsdataframe.to_excel(writer)

    #Store list of symbols for stocks with data errors
    
    FailureListStorage = open('FailureList.txt','w')
    FailureListStorage.write(str(list_failedstocks))
    FailureListStorage.close()

except:
    print("Error when logging results to spreadsheet")

'''

# Close connection to avoid interference with other scripts using same DB




#------------------------- Not in use ------------------------

'''
    #df["PriceChangeFlag"] = np.where(PriceChangeParam,1,0)
    #df["DollarVolFlag"] = np.where(DollarVolParam,1,0)
    #df["SharePriceFlag"] = np.where(SharePriceParam,1,0)


    #df.insert(0, "Symbol", stock) #add colum for 'symbol' [no longer relevant]
'''

