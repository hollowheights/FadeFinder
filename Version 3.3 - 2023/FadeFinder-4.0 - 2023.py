#--- This version is started on 9.02.2023, first update to the program since summer 2022 ---
#Major changes: now using postgres instead of SQLite+feather - reflected in ticker list creation and in load of data

# 1 --- IMPORTS AND SIMILAR ---#
import pandas as pd
import numpy as np
import time
import datetime
from dateutil import tz
from pandas.tseries.offsets import BDay #used in intraday func
import sqlalchemy
from sqlalchemy import create_engine, text
import pickle
import logging
import concurrent.futures
#from scipy.stats.mstats import gmean
import cProfile
import pstats

logging.basicConfig(level=logging.CRITICAL) # possible values include: "DEBUG", "CRITICAL" (always in caps)
pd.set_option("display.max_rows", 1000, "display.min_rows", 200, "display.max_columns", None, "display.width", None)
pd.options.mode.chained_assignment = None  # default='warn'

def timezone_convert(time):
    return time.replace(tzinfo=tz.UTC).astimezone(tz.gettz('America/New_York'))

# 2 -------------- Connect to local postgresql db - Set up test universe - --------------------
try:
    engine = sqlalchemy.create_engine('postgresql+psycopg2://postgres:postgres@localhost:5432/US_Listed')
except:
    logging.critical("Database not accessible currently")

# Define ticker_list from pickle file + sort it so the order is unchanged between runs of this script
path = r"C:\Users\Simon\OneDrive\01 Data Partition\04 Programmering\04 Python Programmering\PycharmProjects\SR-Polygon\SR-Downloading-Scripts\tickerlist.pkl"
with open(path, "rb") as f:
    ticker_list = list(pickle.load(f))
    ticker_list.sort()


# 3 -------------------- Parameters for the backtest---------------------
#used to have both startdate and enddate in string form - why?

# Note to self: måske lav en funktion til automatisk at finde seneste dato med et setup -> og så altid køre for et
# fast prisinterval -> slipper for at holde styr på hvilke pris intervaller er kørt for hvilke dato-intervaller

startdate = datetime.datetime.strptime("2023-10-12",'%Y-%m-%d').date()
enddate = datetime.date.today()
#enddate = datetime.date.today().strftime('%Y-%m-%d')

Signalperiod = "gap"  #"gap" / "day"
Exposure = 0.1  #percentage of portfolio to use for each trade
Fees = 0.000  #Percentage - 1=100% - Currently 0 and added/adjusted in Jupyter Notebook
direction = "short"  # "short" / "long"

# Indicators: 0=off, 1=on
PriceChangeFilter = 1  # Change from close to open in percentage
PriceChangeMinSetting = 20
PriceChangeMaxSetting = 9999

RVOLfilter = 0  # Relative volume
RVOLSetting = 0

VolumeFilter = 1 # introduces hindsight bias - used as a proxy since prevolume is not avail. in the daily table
VolumeSetting = 200_000

PreVolFilter = 1
PreVolSetting = 200_000

DollarVolfilter = 0  # Absolute volume to secure liquidity
DollarVolSetting = 00_000

PercentileFilter = 0  # Close in percentile of day's range
PercentileSetting = 50

SharePriceFilter = 1  # Exclude the cheapest stocks - possibly redundant when based on adj. data
SharePriceSetting = [0.2, 9999]

MarketCapFilter = 1 # Calculated from close of previous day
MarketCapSetting = 999_000_000


# 4 ------------------- Definition of main functions ----------------------------------

#This functino is called from TradeFunctionDaily when a signal is found
def TradeFunctionIntraday(stock, date): #takes a stock symbol (string) and a datetime instance
    logging.debug(f"Starting intraday function for ticker: {stock}")

    # set up df from intraday postgresql table - load the entire day since we work both with PM and RH
    query= f""" SELECT * FROM intraday WHERE "Stock" = '{stock}' AND
      ("Time" AT TIME ZONE 'US/Eastern')::timestamp::date = '{date}'::date; """

    df = pd.read_sql_query(query, con=engine, parse_dates=True)
    df = df.sort_values(by="Time")

    # convert time zone - defaults to UTC on load
    df.Time = df.Time.dt.tz_convert('America/New_York')

    # Regular hours
    RegularStart = str(date) + " 09:30"
    RegularEnd = str(date) + " 16:00"
    RegularMask = (df['Time'] >= RegularStart) & (df['Time'] < RegularEnd)

    # Premarket hours
    PreMarketStart = str(date) + " 04:00"
    PreMarketEnd = str(date) + " 09:30"
    PreMarketMask = (df['Time'] >= PreMarketStart) & (df['Time'] < PreMarketEnd)

    PremarketTotalVolume = df[PreMarketMask]['Volume'].sum()

    # Check if PreVolume is above threshold - if not the script moves on to the next daily signal
    if PremarketTotalVolume < PreVolSetting:
        return None

    else:
        PremarketHigh = df[PreMarketMask]['High'].max()

        #Break of premarket high | first checking that price breaks PreHigh
        if len(df[RegularMask & (df['High'] > PremarketHigh)]) > 0:
            Time_unformatted = df[RegularMask & (df['High'] > PremarketHigh)]['Time'].iloc[0]
            PreMarketBreakTime = Time_unformatted.time()
        else:
            PreMarketBreakTime = np.nan

        #HOD time
        Rownum = df[RegularMask]['High'].idxmax() # determine the row number for HOD
        HODTime = df.loc[Rownum, "Time"].time()   # extract the time for this row

        #Specific time points - to plot average development through the day (NOTE: open can differ from close T-1)
        #using .sum() for now - should be changed to using .loc and thus avoiding sum()
        Price935 = df[df['Time'] == str(date) + " 9:35"]['Open'].sum()
        Price945 = df[df['Time'] == str(date) + " 9:45"]['Open'].sum()
        Price1000 = df[df['Time'] == str(date) + " 10:00"]['Open'].sum()
        Price1100 = df[df['Time'] == str(date) + " 11:00"]['Open'].sum()
        Price1300 = df[df['Time'] == str(date) + " 13:00"]['Open'].sum()
        Price1500 = df[df['Time'] == str(date) + " 15:00"]['Open'].sum()
        Price1530 = df[df['Time'] == str(date) + " 15:30"]['Open'].sum()
        Price1600 = df[df['Time'] == str(date) + " 16:00"]['Open'].sum()

        return (PremarketTotalVolume, PremarketHigh, PreMarketBreakTime, HODTime,
                Price935,Price945,Price1000,Price1100,Price1300,Price1500, Price1530, Price1600)

resultspartialdataframes = []  # storage list to build 'resultsdataframe' from
failedstocks = []  # storage list symbols of stocks with data errors
failedstocks_intraday = []

def TradeFunctionDaily(stock): # columns in DB Daily -> check excel file for column definitions
    logging.debug(f"starting TradeFunctionDaily for {stock}")

    # Initialize dataframe with daily data -
    query = f"SELECT * FROM daily WHERE UPPER(\"Stock\") = '{stock}' AND \"Date\" BETWEEN '{startdate}' AND '{enddate}'"

    df = pd.read_sql_query(query, con=engine, parse_dates=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by="Date")

    # Define booleans that are invariant between long/short tests
    DollarVolParam = df.Volume.shift(1) > DollarVolSetting
    SharePriceParam = (df.OpenUnadjusted > SharePriceSetting[0]) & (df.OpenUnadjusted < SharePriceSetting[1])
    MarketCapParam = (df.MarketCap.shift(1) < MarketCapSetting) #day 1 can never give a signal -> as it should be
    VolumeParam = (df.Volume > VolumeSetting)

    #Define PriceChangeParam depending on "signalperiod"
    if Signalperiod == "day":
        PriceChangeParam = ((df.Close - df.Open) / df.Range.mean()) > PriceChangeMinSetting
    elif Signalperiod == "gap":
        PriceChangeMinParam = ((((df.Open - df.Close.shift(1)) / df.Close.shift(1)) * 100 ) > PriceChangeMinSetting)
        PriceChangeMaxParam = ((((df.Open - df.Close.shift(1)) / df.Close.shift(1)) * 100 ) < PriceChangeMaxSetting)
        PriceChangeParam = (PriceChangeMinParam & PriceChangeMaxParam)

        # Check that the prior trading day is less than x days back - to avoid day 1's after a stock is relisted
        # This needs adjustments if using signalperiod 'day'
        DaysBreakParam = np.where((df.Date-df.Date.shift(1)) < datetime.timedelta(days=7), 1, 0)

    # Define columns based on the above params -> this is for debugging not for actual filtering (params used directly)
    df["PriceChangeParam"] = np.where((PriceChangeParam), 1, 0)
    df["VolumeParam"] = np.where(VolumeParam, 1, 0)
    df["SharePriceParam"] = np.where(SharePriceParam, 1, 0)
    df["MarketCapParam"] = np.where(MarketCapParam, 1, 0)
    df["DaysBreakParam"] = np.where(DaysBreakParam, 1, 0)

    #Define combined filter condition for tradesignal and flag all days with a signal
    TradeSignalParam = (PriceChangeParam & VolumeParam & SharePriceParam & MarketCapParam & DaysBreakParam) == True
    df["TradeSignal"] = np.where(TradeSignalParam, 1, 0)

    # Calculate basic numbers for signal days - reminder about shift: positive number = back in time (ascending order)
    if Signalperiod == "gap":
        df["GapSize"] = np.where(TradeSignalParam, ((df.Open / df.Close.shift(1)) - 1)*100, 0)
        df["GapSizeRel"] = np.where(TradeSignalParam, df.GapSize / df.Range.mean(), 0) #tilføj evt vægtet gns af range
        df["GapSizeAbs"] = np.where(TradeSignalParam, df.OpenUnadjusted - df.CloseUnadjusted.shift(1),0)
        df["RVOL10D"] = np.where(TradeSignalParam, df.Volume/df.AvgVolume10D,0) #OBS RVOL premarket er mere aktuelt

        df["Day1"] = np.where(TradeSignalParam, (df.CloseUnadjusted - df.OpenUnadjusted) / df.OpenUnadjusted, 0)
        df["Day1/Gap"] = np.where(TradeSignalParam, (df.CloseUnadjusted - df.OpenUnadjusted) / df.GapSizeAbs, 0)
        df["Day1/Gap_adj"] = np.where(TradeSignalParam, (df.Close - df.Open)/(df.Open - df.Close.shift(1)), 0)
        df["Overnight"] = np.where(TradeSignalParam, (df.Close - df.Open.shift(-1)) / df.Open, 0)
        df["Day2"] = np.where(TradeSignalParam, (df.Open.shift(-1) - df.Close.shift(-1)) / df.Open, 0)
        df["2Day"] = np.where(TradeSignalParam, ((df.Open - df.Close.shift(-1)) / df.Open), 0) #day + night + day

        df["MaxGain/Gap"] = np.where(TradeSignalParam,(df.HighUnadjusted - df.OpenUnadjusted) / (df.OpenUnadjusted - df.CloseUnadjusted.shift(1)), 0)
        df["OpenAdjusted"] = df.Open
        df["MarketCap"] = df["MarketCap"].shift(1)
        df["RangeAvg10D"] = df.Range.rolling(10).mean()
        df["TrueRange"] = df[["High","PrevDayClose"]].max(axis=1) - df[["Low","PrevDayClose"]].min(axis=1)
        df['Gap/ATR10'] = np.where(TradeSignalParam, df.GapSizeAbs / df.TrueRange.shift(1).rolling(10).mean(), 0)

    # Set values for trades based on the above - [no indentation since it works for both Signal periods (day/gap)]
    df["Day1Trade"] = np.where(TradeSignalParam, -df['Day1/Gap'], 0) #temp. set to this simple measure - needs fixing
    #df["Day1Trade"] = np.where(TradeSignalParam, (df["Day1"]-Fees) * Exposure,0)
    df["Day2Trade"] = np.where(TradeSignalParam, (df["Day2"]-Fees) * Exposure,0)
    df["OvernightTrade"] = np.where(TradeSignalParam, (df["Overnight"] -Fees)* Exposure,0)
    df["2DayTrade"] = np.where(TradeSignalParam, (df["2Day"] - Fees) * Exposure, 0)

    #To inspect the result of each parameter, check the full dataframe - currently disabled
    #logging.debug(df)

    #Set up a new dataframe for only the days giving a trade signal
    df_tradesignals = df[df.TradeSignal == 1]

    #Add intraday data
    if len(df_tradesignals) > 0:
        logging.debug(f"There are trade signals")

        try:
            #Call intraday function and store the return value(s)

            intraday_col_names = ['PreVolume','PreHigh','PreBreakTime','HODTime','Price935','Price945','Price1000','Price1100','Price1300','Price1500', 'Price1530', 'Price1600']

            # x represents each row
            df_tradesignals[intraday_col_names] = df_tradesignals.apply(lambda x: TradeFunctionIntraday(x['Stock'], x['Date']) or [None]*12, axis=1, result_type='expand')

            # filter out tradesignals with insufficient premarket volume
            df_tradesignals = df_tradesignals[df_tradesignals["PreVolume"].notnull()]

            # Calculating the column here - when prev day is still avail. -  more robust when doing it based on the daily data than intraday
            df_tradesignals['Open/PreHigh'] = df_tradesignals['GapSize'] / (((df_tradesignals['PreHigh'] / df_tradesignals['PrevDayClose']) - 1) * 100)

            df_tradesignals['PreRVOL'] = df_tradesignals['PreVolume'] / df_tradesignals[
                "AvgVolume10D"]  # obs denne bruger en kolonne der ikke optræder eksplicit her i scriptet

            # Add all trading signals to temporary storage
            resultspartialdataframes.append(df_tradesignals)

        except Exception as e:
            logging.critical(e)
            logging.critical(f"failed to add intraday data for stock {stock}")
            failedstocks_intraday.append(stock)

    else:
        logging.debug(f"No trade signals for stock: {stock}")
        pass

# 5 ------------------------- Initiate the backtest ------------------------

# Testing partial list is effective for testing and for runtime estimation
ticker_list = ticker_list[0:-1]
#ticker_list = ["SNOA"] #ACMR

timemeasure = time.perf_counter()

with concurrent.futures.ThreadPoolExecutor(5) as executor:  # 5 threads seem to be optimal on my Thinkpad
    executor.map(TradeFunctionDaily, ticker_list)

#cProfile - not in use currently
# with cProfile.Profile() as pr:
#     for result in map(TradeFunctionDaily, ticker_list):
#         pass  # You can perform any operation with the result here if needed

# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME)
# stats.print_stats()


# 6 ---------------------Set up  dataframe for results----------------------------

#create 'resultsdataframe'
if len(resultspartialdataframes) > 0:
    resultsdataframe = pd.concat(resultspartialdataframes, ignore_index=True)
elif (len(resultspartialdataframes)) < 1:
    print("There are no instances in the analysis results. Exiting script early.")
    exit()
else:
    resultsdataframe = resultspartialdataframes[0]

#Remove instances with too low PreVolume - they are all currently set to NaN
#this can be skipped if/when the intraday function is changed to exit tickers with too low PreVolume
#if resultsdataframe
resultsdataframe = resultsdataframe[~resultsdataframe.PreVolume.isna()]

#Sort resultsdataframe by date - to reflect correlation between stocks (for equity curve)
resultsdataframe.sort_values(by=["Date"],ascending=True,inplace=True)
resultsdataframe.reset_index(inplace=True,drop=True)

# Add 4 columns with cumulative results
resultsdataframe["Day1TradeCum"] = resultsdataframe["Day1Trade"].cumsum()
resultsdataframe["Day2TradeCum"] = resultsdataframe["Day2Trade"].cumsum()
resultsdataframe["OvernightTradeCum"] = resultsdataframe["OvernightTrade"].cumsum()
resultsdataframe["2DayTradeCum"] = resultsdataframe["2DayTrade"].cumsum()

# Add columns for change from open relative to gap size - (negative here means positive return for shorting)
resultsdataframe['Change935/Gap'] = np.where(resultsdataframe['Price935']>0,(resultsdataframe['Price935'] - resultsdataframe['OpenUnadjusted']) / resultsdataframe['GapSizeAbs'], np.nan)
resultsdataframe['Change945/Gap'] = np.where(resultsdataframe['Price945']>0,(resultsdataframe['Price945'] - resultsdataframe['OpenUnadjusted']) / resultsdataframe['GapSizeAbs'], np.nan)
resultsdataframe['Change1000/Gap'] = np.where(resultsdataframe['Price1000']>0,(resultsdataframe['Price1000'] - resultsdataframe['OpenUnadjusted']) / resultsdataframe['GapSizeAbs'], np.nan)
resultsdataframe['Change1100/Gap'] = np.where(resultsdataframe['Price1100']>0,(resultsdataframe['Price1100'] - resultsdataframe['OpenUnadjusted']) / resultsdataframe['GapSizeAbs'], np.nan)
resultsdataframe['Change1300/Gap'] = np.where(resultsdataframe['Price1300']>0,(resultsdataframe['Price1300'] - resultsdataframe['OpenUnadjusted']) / resultsdataframe['GapSizeAbs'], np.nan)
resultsdataframe['Change1500/Gap'] = np.where(resultsdataframe['Price1500']>0,(resultsdataframe['Price1500'] - resultsdataframe['OpenUnadjusted']) / resultsdataframe['GapSizeAbs'], np.nan)
resultsdataframe['Change1530/Gap'] = np.where(resultsdataframe['Price1530']>0,(resultsdataframe['Price1530'] - resultsdataframe['OpenUnadjusted']) / resultsdataframe['GapSizeAbs'], np.nan)
resultsdataframe['Change1600/Gap'] = np.where(resultsdataframe['Price1600']>0,(resultsdataframe['Price1600'] - resultsdataframe['OpenUnadjusted']) / resultsdataframe['GapSizeAbs'], np.nan)

# Delete unnecessary columns + renaming
resultsdataframe.drop(["Open","TradeSignal"], axis=1, inplace=True)
resultsdataframe.rename({}, axis=1, inplace=True)

#Set order of resultsdataframe columns
indexlist = ["Date","Stock","GapSize","MarketCap","OpenAdjusted","OpenUnadjusted","PreVolume", "AvgVolume10D","PreRVOL", "Pre$Volume","PreHigh", 'Open/PreHigh', 'Weekday', "GapSizeAbs",'GapSizeRel',
             "Range","RangeAvg10D","TrueRange",'Gap/ATR10',
             "MaxGain/Gap", "PreBreakTime","HODTime","Day1/Gap","Day1/Gap_adj", "Day1","Day2","Overnight","2Day","Volume","$volume","Close","High","Low",'CloseUnadjusted', 'HighUnadjusted','LowUnadjusted',
             "Day1Trade","Day2Trade","OvernightTrade","2DayTrade","Day1TradeCum","Day2TradeCum","OvernightTradeCum","2DayTradeCum",
             "AF6M","AF12M",
             'Price935', 'Price945', 'Price1000','Price1100','Price1300','Price1500','Price1530','Price1600',
             "Change935/Gap","Change945/Gap","Change1000/Gap","Change1100/Gap","Change1300/Gap","Change1500/Gap","Change1530/Gap","Change1600/Gap"]

resultsdataframe = resultsdataframe.reindex(indexlist, axis=1)


#connect.close()  # Close connection to avoid interference with other scripts using same DB

# ---------------------------------------------------------------------------------------------------------------
#                  At this point all that's left: presenting data in terminal and logging to file               #
# ---------------------------------------------------------------------------------------------------------------


# 7 --------------------- Preparing data for presentation ----------------------------------------------------------------

try:
    Logdataframe = pd.DataFrame({
        "Test date": [datetime.date.today()],
        "Direction": [direction],
        "# Stocks": [len(ticker_list)],
        "# Days": [np.busday_count(startdate, enddate)],
        "Total rows": [np.busday_count(startdate, enddate) * len(ticker_list)],
        "Trade signals": [resultsdataframe.shape[0]],
        "Signal frequency": ["1 / %s" % ((np.busday_count(startdate, enddate) * len(ticker_list)) / resultsdataframe.shape[0])],
        "RVOL setting": [RVOLSetting],
        "Price change minimum % setting": [PriceChangeMinSetting],
        "Range percentile setting": [PercentileSetting] })

except Exception as e:
    logging.debug("Logdataframe could not be defined")
    logging.debug(e)


# 8 --------------------------------- Results analysis ------------------------------

#Print first 20 lines of resultsdataframe
print("resultsdataframe: \n",resultsdataframe.tail(15))

#Print the dataframe holding parameter settings and results
print("\n", Logdataframe.to_string(index=False))

#Stats on stocks failing
print("Failure list's length now: ", len(failedstocks))

#Runtime of script - placed before plots due to blocking tendencies
print("Time to run script: ", time.perf_counter()-timemeasure)


# 9 ------------------------------ Logging to files  -------------------------------------------------

try:
    #Testresults added to previous results
    #pd.DataFrame.to_csv(Logdataframe,r"C:\ Users\LENOVO\desktop\ testfilmarts2.csv",mode="a",header=True,index=False) #OBS fjern mellemrum i path!

    #Store latest instance of resultsdataframe temporary - useful when comparing versions
    #current_time = str(datetime.datetime.now)
    current_time = datetime.datetime.now().strftime("%d.%m.%Y-%H.%M.%S")

    # needs to backslashes at the end - to escape the final backslash
    folder_path = r"C:\Users\Simon\OneDrive\01 Data Partition\04 Programmering\04 Python Programmering\PycharmProjects\FadeFinder\Output filer\\"

    with pd.ExcelWriter(f"{folder_path}FadeFinderOutput_{current_time}.xlsx") as writer:
        resultsdataframe.to_excel(writer, index=False)

    #Store list of symbols for stocks with data errors

    '''
    FailureListStorage = open('FailureList.txt','w')
    FailureListStorage.write(str(failedstocks))
    FailureListStorage.close()
    '''
except Exception as e:
    print("Error when logging resultsdataframe to xlsx")
    print(e)


# 10 ----------------------------- Basic plotting -------------------------------------

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

# 11 ------------------------- Misc ------------------------

'''
#uvist hvorfor dette ikke virkede længere - måske fordi jeg havde ændret data type i daily tabellen?
df = df.sort_values(by="Date")

# Apply test period settings
df = df[(df['Date'] >= startdate) & (df['Date'] <= enddate)]

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
