import pandas as pd

class ReadFile:

    def __init__(self) -> None:
        pass

    def Read_Excel(self,file_path):
        return pd.read_excel(file_path) #return type is Data Frame