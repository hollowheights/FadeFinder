# 1 --- IMPORTS AND SIMILAR ---#
import pandas as pd
import numpy as np
import sqlite3
import time
import datetime
from dateutil import tz
from pandas.tseries.offsets import BDay #used in intraday func
from scipy.stats.mstats import gmean
pd.set_option("display.max_rows", 1000, "display.min_rows", 200, "display.max_columns", None, "display.width", None)
pd.options.mode.chained_assignment = None  # default='warn'
timemeasure = time.perf_counter()
def timezone_convert(time): return time.replace(tzinfo=tz.UTC).astimezone(tz.gettz('America/New_York'))


# 2 -------------- Connect to local SQLite db - Set up test universe - --------------------
connect = sqlite3.connect(r"C:\01 SR Data partition\TradingDatabaser\SRUSListedPolygonDaily.db")
c = connect.cursor()

# add all symbols in the database to 'stocklist'
stocklist = []
c.execute('''select * FROM sqlite_master WHERE type="table"''')
storagelist = c.fetchall()

for x in range(len(storagelist)):
    stocklist.append(storagelist[x][1])

stocklist = stocklist[450:650 ] #Limit test to a subsection of the test universe

# 3 -------------------- Parameters for the backtest---------------------
startdate = "2022-01-01"
enddate = "2022-03-05"
Signalperiod = "gap"  #"gap" / "day"
Exposure = 0.1  #percentage of portfolio to use for each trade
Fees = 0.000  #Percentage - 1=100% - Currently 0 and added/adjusted in Jupyter Notebook
direction = "short"  # "short" / "long"

# 0=off, 1=on
PriceChangeFilter = 1  # Change from close to open in percentage
PriceChangeSetting = 10

RVOLfilter = 0  # Relative volume
RVOLSetting = 2

# Overvej i stedet/yderligere at filtrere på premarket volume > x -> bør være mere direkte relevant
DollarVolfilter = 0  # Absolute volume to secure liquidity
DollarVolSetting = 10_000

PercentileFilter = 0  # Close in percentile of day's range
PercentileSetting = 50

SharePriceFilter = 0  # Exclude the cheapest stocks - possibly redundant when based on adj. data
SharePriceSetting = 0

MarketCapFilter = 1 # Calculated from close of previous day
MarketCapSetting = 1200_000_000


# 4 ------------------- Definition of main functions ----------------------------------

def TradeFunctionIntraday(stock,date):
    #Receives a stock symbol (string) and a date (datetime instance)
    #stock = stock.lower()

    # set up df from intraday data (feather) - time column in df is datetime64
    df = pd.read_feather(r"C:\01 SR Data partition\TradingDatabaser\SRFeather\SRUSListedPolygonIntraday\%s.feather" %stock)

    # establish a df for 3 days to reduce data processing + adjust timezone
    day_before = datetime.datetime.combine (date - BDay(1), datetime.datetime.min.time())
    day_after = datetime.datetime.combine (date + BDay(1), datetime.datetime.max.time())
    df_3day = df[(df['time'] >= day_before) & (df['time'] <= day_after)].copy()
    df_3day['time'] = df_3day.loc[:, 'time'].map(lambda x: timezone_convert(x))

    # Regular hours
    RegularStart =  str(date) + " 09:30"
    RegularEnd = str(date) + " 16:00"
    RegularMask = (df_3day['time'] >= RegularStart) & (df_3day['time'] < RegularEnd)

    # Premarket hours
    PreMarketStart = str(date) + " 04:00"
    PreMarketEnd = str(date) + " 09:30"
    PreMarketMask = (df_3day['time'] >= PreMarketStart) & (df_3day['time'] < PreMarketEnd)

    PremarketTotalVolume = df_3day[PreMarketMask]['volume'].sum()
    PremarketHigh = df_3day[PreMarketMask]['high'].max()

    #Break of premarket high | first checking that price breaks PreHigh
    if len(df_3day[RegularMask & (df_3day['high'] > PremarketHigh)]) > 0:
        Time_unformatted = df_3day[RegularMask & (df_3day['high'] > PremarketHigh)]['time'].iloc[0]
        PreMarketBreakTime = Time_unformatted.time()
    else:
        PreMarketBreakTime = np.nan

    #HOD time
    Rownum = df_3day[RegularMask]['high'].idxmax() #1 Bestem rækken hvor HOD findes
    HODTime = df_3day.loc[Rownum,"time"].time()    #2 udtræk tidspunktet for denne række

    #Specific time points - to plot average development through the day
    Price935 = df_3day[df_3day['time'] == str(date) + " 9:35"]['close'].sum()
    Price945 = df_3day[df_3day['time'] == str(date) + " 9:45"]['close'].sum()
    Price1000 = df_3day[df_3day['time'] == str(date) + " 10:00"]['close'].sum()
    Price1100 = df_3day[df_3day['time'] == str(date) + " 11:00"]['close'].sum()
    Price1300 = df_3day[df_3day['time'] == str(date) + " 13:00"]['close'].sum()
    Price1500 = df_3day[df_3day['time'] == str(date) + " 15:00"]['close'].sum()

    return (PremarketTotalVolume, PremarketHigh, PreMarketBreakTime, HODTime, Price935,Price945,Price1000,Price1100,Price1300,Price1500)


resultspartialdataframes = []  # storage list to build 'resultsdataframe' from
failedstocks = []  # storage list symbols of stocks with data errors
failedstocks_intraday = []

def TradeFunctionDaily(stock):
    # columns in DB Daily -> check excel file for info
    # Initialize dataframe with daily data
    df = (pd.read_sql("SELECT * FROM %s WHERE DATE(date) BETWEEN '%s' AND '%s' " % (stock, startdate, enddate), connect, parse_dates=['Date']) )

    # Define booleans for checks that are invariant between long/short test
    DollarVolParam = df.Volume.shift(1) > DollarVolSetting
    SharePriceParam = df.OpenUnadjusted > SharePriceSetting
    MarketCapParam = (df.MarketCap.shift(1) < MarketCapSetting)

    #Define PriceChangeParam depending on "signalperiod"
    if Signalperiod == "day":
        PriceChangeParam = ((df.Close - df.Open) / df.Range.mean()) > PriceChangeSetting
    elif Signalperiod == "gap":
        PriceChangeParam = (((df.Open - df.Close.shift(1)) / df.Close.shift(1)) * 100 ) > PriceChangeSetting
        DaysBreakParam = np.where(PriceChangeParam,(df.Date-df.Date.shift(1)) < datetime.timedelta(days=5),False)

    #Define combined filter condition for tradesignal and flag all days with a signal
    TradeSignalParam = (PriceChangeParam & MarketCapParam & DaysBreakParam) == True
    df["TradeSignal"] = np.where(TradeSignalParam,1,0)

    # Calculate basic numbers for signal days
    #shift: positive tal for at gå tilbage i tid, neg. tal for at gå frem i tid (ascending order)
    if Signalperiod == "gap":

        df["GapSize"] = np.where(TradeSignalParam, ((df.Open / df.Close.shift(1)) - 1)*100, 0)
        df["GapSizeRel"] = np.where(TradeSignalParam, df.GapSize / df.Range.mean(), 0) #tilføj evt vægtet gns af range
        df["GapSizeAbs"] = np.where(TradeSignalParam, df.OpenUnadjusted - df.CloseUnadjusted.shift(1),0)
        df["RVOL10D"] = np.where(TradeSignalParam,df.Volume/df.AvgVolume10D,0) #OBS RVOL premarket er mere aktuelt

        df["Day1"] = np.where(TradeSignalParam, (df.OpenUnadjusted - df.CloseUnadjusted) / df.OpenUnadjusted, 0)
        df["Day1/Gap"] = np.where(TradeSignalParam, (df.OpenUnadjusted-df.CloseUnadjusted)/(df.OpenUnadjusted-df.CloseUnadjusted.shift(1)), 0)
        df["Day1/Gap_adj"] = np.where(TradeSignalParam, (df.Open-df.Close)/(df.Open-df.Close.shift(1)), 0)
        df["Overnight"] = np.where(TradeSignalParam, (df.Close - df.Open.shift(-1)) / df.Open, 0)
        df["Day2"] = np.where(TradeSignalParam, (df.Open.shift(-1) - df.Close.shift(-1)) / df.Open, 0)
        df["2Day"] = np.where(TradeSignalParam, ((df.Open - df.Close.shift(-1)) / df.Open), 0) #day + night + day
        df["MaxGain/Gap"] = np.where(TradeSignalParam,(df.HighUnadjusted - df.OpenUnadjusted) / (df.OpenUnadjusted - df.CloseUnadjusted.shift(1)), 0)
        df["OpenAdjusted"] = df.Open
        df["MarketCap"] = df["MarketCap"].shift(1)
        df["RangeAvg10D"] = df.Range.rolling(10).mean()
        df["TrueRange"] = df[["High","PrevDayClose"]].max(axis=1) - df[["Low","PrevDayClose"]].min(axis=1)
        df['Gap/ATR10'] = np.where(TradeSignalParam, df.GapSizeAbs / df.TrueRange.shift(1).rolling(10).mean(), 0)

    # Set values for trades based on the above - [no indentation since it works for both Signal period = day/gap]
    df["Day1Trade"] = np.where(TradeSignalParam, (df["Day1"]-Fees) * Exposure,0)
    df["Day2Trade"] = np.where(TradeSignalParam, (df["Day2"]-Fees) * Exposure,0)
    df["OvernightTrade"] = np.where(TradeSignalParam, (df["Overnight"] -Fees)* Exposure,0)
    df["2DayTrade"] = np.where(TradeSignalParam, (df["2Day"] - Fees) * Exposure, 0)

    df_tradesignals = df[df.TradeSignal == 1]

    # Add premarket data
    PreStorageTotalVolume = []
    PreStorageHigh = []
    PreStorageBreakTime = []
    PreStorageHODTime = []
    PreStoragePrice935 = []
    PreStoragePrice945 = []
    PreStoragePrice1000 = []
    PreStoragePrice1100 = []
    PreStoragePrice1300 = []
    PreStoragePrice1500 = []

    for row in df_tradesignals.index:
        try:
            responses = TradeFunctionIntraday(df_tradesignals.loc[row, "Stock"], df_tradesignals.loc[row, "Date"])
            PreStorageTotalVolume.append(responses[0])
            PreStorageHigh.append(responses[1])
            PreStorageBreakTime.append(responses[2])
            PreStorageHODTime.append(responses[3])
            PreStoragePrice935.append(responses[4])
            PreStoragePrice945.append(responses[5])
            PreStoragePrice1000.append(responses[6])
            PreStoragePrice1100.append(responses[7])
            PreStoragePrice1300.append(responses[8])
            PreStoragePrice1500.append(responses[9])

        except:
            PreStorageHigh.append(np.nan)
            PreStorageTotalVolume.append(np.nan)
            PreStorageBreakTime.append(np.nan)
            PreStorageHODTime.append(np.nan)
            PreStoragePrice935.append(np.nan)
            PreStoragePrice945.append(np.nan)
            PreStoragePrice1000.append(np.nan)
            PreStoragePrice1100.append(np.nan)
            PreStoragePrice1300.append(np.nan)
            PreStoragePrice1500.append(np.nan)

    df_tradesignals['PreVolume'] = PreStorageTotalVolume
    df_tradesignals['PreHigh'] = PreStorageHigh
    df_tradesignals['PreBreakTime'] = PreStorageBreakTime
    df_tradesignals['HODTime'] = PreStorageHODTime
    df_tradesignals['Price935'] = PreStoragePrice935
    df_tradesignals['Price945'] = PreStoragePrice945
    df_tradesignals['Price1000'] = PreStoragePrice1000
    df_tradesignals['Price1100'] = PreStoragePrice1100
    df_tradesignals['Price1300'] = PreStoragePrice1300
    df_tradesignals['Price1500'] = PreStoragePrice1500


    #Calculating the column here - when prev day is still avail. -  more robust when doing it based on the daily data than intraday
    df_tradesignals['Open/PreHigh'] = df_tradesignals['GapSize'] / ( ((df_tradesignals['PreHigh']/df_tradesignals['PrevDayClose'])-1)*100 )
    df_tradesignals['PreRVOL'] = df_tradesignals['PreVolume'] / df_tradesignals["AvgVolume10D"] #obs denne bruger en kolonne der ikke optræder eksplicit her i scriptet

    # Add all trading signals to temporary storage
    resultspartialdataframes.append(df_tradesignals)


# 5 ------------------------- Initiate the backtest ------------------------
for stock in stocklist[0:-1]:
    try:
        TradeFunctionDaily(stock)
    except:
        print("Error with stock: %s" % stock)
        failedstocks.append(stock)


# 6 ---------------------Set up  dataframe for results----------------------------

#create 'resultsdataframe'
if len(resultspartialdataframes) > 1:
    resultsdataframe = pd.concat(resultspartialdataframes, ignore_index=True)
else:
    resultsdataframe = resultspartialdataframes[0]

#Sort resultsdataframe by date - to reflect correlation between stocks (for equity curve)
resultsdataframe.sort_values(by=["Date"],ascending=True,inplace=True)
resultsdataframe.reset_index(inplace=True,drop=True)

# Add 4 columns with cumulative results
resultsdataframe["Day1TradeCum"] = resultsdataframe["Day1Trade"].cumsum()
resultsdataframe["Day2TradeCum"] = resultsdataframe["Day2Trade"].cumsum()
resultsdataframe["OvernightTradeCum"] = resultsdataframe["OvernightTrade"].cumsum()
resultsdataframe["2DayTradeCum"] = resultsdataframe["2DayTrade"].cumsum()

# Add columns for change from open relative to gap size at specific times - negative here means positive return for shorting
GapSize = (resultsdataframe['OpenAdjusted'] - resultsdataframe['PrevDayClose'])
resultsdataframe['Change935/Gap'] = np.where(resultsdataframe['Price935']>0,(resultsdataframe['Price935'] - resultsdataframe['OpenAdjusted']) / GapSize, np.nan)
resultsdataframe['Change945/Gap'] = np.where(resultsdataframe['Price945']>0,(resultsdataframe['Price945'] - resultsdataframe['OpenAdjusted']) / GapSize, np.nan)
resultsdataframe['Change1000/Gap'] = np.where(resultsdataframe['Price1000']>0,(resultsdataframe['Price1000'] - resultsdataframe['OpenAdjusted']) / GapSize, np.nan)
resultsdataframe['Change1100/Gap'] = np.where(resultsdataframe['Price1100']>0,(resultsdataframe['Price1100'] - resultsdataframe['OpenAdjusted']) / GapSize, np.nan)
resultsdataframe['Change1300/Gap'] = np.where(resultsdataframe['Price1300']>0,(resultsdataframe['Price1300'] - resultsdataframe['OpenAdjusted']) / GapSize, np.nan)
resultsdataframe['Change1500/Gap'] = np.where(resultsdataframe['Price1500']>0,(resultsdataframe['Price1500'] - resultsdataframe['OpenAdjusted']) / GapSize, np.nan)

#Delete unnecessary columns + renaming
resultsdataframe.drop(["Open","TradeSignal"
                       ], axis=1, inplace=True)
resultsdataframe.rename({}, axis=1, inplace=True)

#Set order of resultsdataframe columns
indexlist = ["Date","Stock","GapSize","MarketCap","OpenAdjusted","OpenUnadjusted","PreVolume", "AvgVolume10D","PreRVOL", "Pre$Volume","PreHigh", 'Open/PreHigh', 'Weekday', "GapSizeAbs",'GapSizeRel',
             "Range","RangeAvg10D","TrueRange",'Gap/ATR10',
             "MaxGain/Gap", "PreBreakTime","HODTime","Day1/Gap","Day1/Gap_adj", "Day1","Day2","Overnight","2Day","Volume","$volume","Close","High","Low",'CloseUnadjusted', 'HighUnadjusted','LowUnadjusted',
             "Day1Trade","Day2Trade","OvernightTrade","2DayTrade","Day1TradeCum","Day2TradeCum","OvernightTradeCum","2DayTradeCum",
             "AF6M","AF12M",
             'Price935', 'Price945', 'Price1000','Price1100','Price1300','Price1500',
             "Change935/Gap","Change945/Gap","Change1000/Gap","Change1100/Gap","Change1300/Gap","Change1500/Gap"]


resultsdataframe = resultsdataframe.reindex(indexlist, axis=1)


connect.close()  # Close connection to avoid interference with other scripts using same DB


# ---------------------------------------------------------------------------------------------------------------
#                  At this point all data should be the same for version 1 and version 2                       #
# ---------------------------------------------------------------------------------------------------------------


# 7 --------------------- Preparing data for presentation ----------------------------------------------------------------

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
        "Price change % setting": [PriceChangeSetting],
        "Range percentile setting": [PercentileSetting] })

except:
    "Error setting up 'Logdataframe'"


# 8 --------------------------------- Results analysis ------------------------------

#Print first 20 lines of resultsdataframe
print("resultsdataframe: \n",resultsdataframe.head(20))

#Print the dataframe holding parameter settings and results
print("\n",Logdataframe.to_string(index=False))

#Stats on stocks failing
print("Failure list's length now: ", len(failedstocks))

#Runtime of script - placed before plots due to blocking tendencies
print("Time to run script: ", time.perf_counter()-timemeasure)


# 9 ----------------------------- Basic plotting -------------------------------------

'''

#Plot the 3 cumulative measures
resultsdataframe["Day1TradeCum"].plot(legend="Day1TradeCum")
resultsdataframe["Day2TradeCum"].plot(legend="day2TradeCum",color="red")
resultsdataframe["OvernightTradeCum"].plot(legend="OvernightTradeCum",color="grey")
resultsdataframe["2DayTradeCum"].plot(legend = "2DayTradeCum",color="black")
#plt.show()

#resultsdataframe.plot(x='GapSize',y="Day1", legend= "trade return", style="o")
#plt.axhline(y=0)
#plt.show() # PLOTTING TEMP. DISABLED

#resultsdataframe.Day1Trade.plot.hist(by=resultsdataframe.RVOL10D,bins=50,grid=True, xlabel="Testio")

#resultsdataframe[resultsdataframe.Day1Trade > 0].count()
#df = resultsdataframe[(resultsdataframe["ChangeRel"] > 10) & (resultsdataframe["Change"]<3)]

'''

# 10 ------------------------------ Logging to files  -------------------------------------------------

'''
try:
    #Testresults added to previous results
    #pd.DataFrame.to_csv(Logdataframe,r"C:\ Users\LENOVO\desktop\ testfilmarts2.csv",mode="a",header=True,index=False) #OBS fjern mellemrum i path!

    #Store latest instance of resultsdataframe temporary - useful when comparing versions
    with pd.ExcelWriter("resultsdataframe.xlsx") as writer:
        resultsdataframe.to_excel(writer)

    #Store list of symbols for stocks with data errors
    
    FailureListStorage = open('FailureList.txt','w')
    FailureListStorage.write(str(failedstocks))
    FailureListStorage.close()

except:
    print("Error when logging results to spreadsheet")
'''


# 11 ------------------------- Misc ------------------------

'''
df = pd.DataFrame({'ratio':np.random.rand(100), 'feet': np.random.rand(100)*10})
print(df)
df.groupby(pd.cut(df.ratio, np.linspace(0,1,11))).feet.mean().plot.bar()
plt.show()

# From Premarket function:
# Using .sum() here - otherwise it will pass index + column value | (BDay = business days, see import)
    CloseDayBefore = df_3day[df_3day['time'] == (str(date - BDay(1)) + " 15:59")]['close'].sum()
    print("CloseDayBefore: ", CloseDayBefore)
    PremarketHighChange = ((PremarketHigh /  CloseDayBefore ) - 1) *100


'''
