from Data.data import Data
from matplotlib.collections import Collection
import pymongo

class Mongodb(Data):

    def __init__(self):
        print("")
    
    def read_Mongo_DB(self):
        print("reading collections from mongodb...")
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        mydb = myclient["Travel-Advisor"]
        mycol = mydb["Place"]
        data =list(mycol.find({"name":"Topkapi Palace"}))
        if len(data)!=0:
            #type(data[0])= dict
            return data
        else: print("Reading operation is not correct from mongodb database... You must control your fields name in query!!!")
      
    
   

    
