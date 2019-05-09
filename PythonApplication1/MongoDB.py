from pymongo import MongoClient
import configparser
# connection

cfg = "./config.cfg"
config_raw = configparser.RawConfigParser()
config_raw.read(cfg)
defaults = config_raw.defaults()
connString = config_raw.get('MongoDB', 'connstring')
database = config_raw.get('MongoDB', 'database')
stockListCollection = config_raw.get('MongoDB', 'stockListCollection')
stockInfoCollection = config_raw.get('MongoDB', 'stockInfoCollection')
stockPredictResultCollection = config_raw.get('MongoDB', 'stockPredictResultCollection')
stockParserLogCollection = config_raw.get('MongoDB', 'stockParserLogCollection')

if connString == '':
	conn = MongoClient()
else:
	conn = MongoClient(connString)

db = conn[database]

def InsertStockList(stockList):	
	collection = db[stockListCollection]
	collection.insert_many(stockList)
	
def InsertStockPredictResult(stockPredictResultList):	
	collection = db[stockPredictResultCollection]
	collection.insert_many(stockPredictResultList)

def InsertOrUpdateStockInfo(stockInfoList):
	collection = db[stockInfoCollection]
	collection.insert_many(stockInfoList)

def InsertParserLog(log):
	collection = db[stockParserLogCollection]
	collection.update({ "Date": log.get("Date") },log,True)

def GetAllLog():
	collection = db[stockParserLogCollection]
	return collection.find({})

def GetStockInfoList(stockNo):
	collection = db[stockInfoCollection]
	return collection.find({ "No": stockNo })
	
def GetStockList():
	collection = db[stockListCollection]
	return collection.find({})