import pandas as pd
import heapq


class ReadFile:

    file_path = ""

    def __init__(self, file_path):
        self.file_path = file_path

    def Read_Excel_Rated(self, filename, city_name):

        return pd.read_excel(
            self.file_path + filename, sheet_name=city_name + "_rated", index_col=0
        )  # return type is Data Frame

    def read_excel_for_flask(self, filename, city_name, place_name):
        try:
            city_data = pd.read_excel(
                self.file_path + filename, sheet_name=city_name
            ).values.tolist()
            place_index = None
            for index, city in enumerate(city_data):
                if city[0].lower().split(" ") == place_name.lower().split(
                    " "
                ) or place_name.lower().__eq__(city_data[0][0].lower()):
                    place_index = index
            if place_index:
                place_data = city_data[place_index][1:]
                top_ten_indices = heapq.nlargest(
                    11, range(len(place_data)), place_data.__getitem__
                )
                top_ten_names = []
                for idx in top_ten_indices:
                    top_ten_names.append(city_data[idx][0])
                top_ten_names = top_ten_names[1:]
                return {
                    "success": True,
                    # "place_name": place_name,
                    # "top_ten_indices": top_ten_indices,
                    "top_ten_names": top_ten_names,
                    # "place_data": place_data,
                }
            return {"success": False}
        except Exception:
            return {"success": False}
