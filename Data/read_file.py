import pandas as pd
from regex import F

class ReadFile:

    file_path = ""

    def __init__(self,file_path):
        self.file_path = file_path

    def Read_Excel_Rated(self,filename,city_name):

        return pd.read_excel(self.file_path+filename,sheet_name=city_name+"_rated",index_col=0) #return type is Data Frame
         
