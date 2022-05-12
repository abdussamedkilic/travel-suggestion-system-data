from Data.data import Data
from matplotlib.collections import Collection
import pymongo

class Mongodb(Data):

    def __init__(self):
        print("")
    
    def read_Mongo_DB(self,city_name):
        """
        city_name must be String.
        
        """
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        mydb = myclient["Travel-Advisor"]
        mycol = mydb["Place"]

        data_list =list(mycol.find({"location":city_name}))
        placeName_list = []
        imageUrl_list = []
        detail_list = []
        comments = []

        if len(data_list)!=0:
            #type(data[0])= dict
            for i in range(0,len(data_list)):
                placeName_list.append(data_list[i].get('name'))
                imageUrl_list.append(data_list[i].get('image'))
                detail_list.append(data_list[i].get('detail'))
                comments.append(data_list[i].get('comments'))

            return comments,[placeName_list],[imageUrl_list],[detail_list]
        else: 
            raise RuntimeError("You must control your fields name in query!!!")
      
    
   

    
