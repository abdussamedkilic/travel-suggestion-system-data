from openpyxl import Workbook
import xlsxwriter

class WriteExcel:
    def __init__(self) -> None:
        pass
    
    def writeExcel_Doc2vec(self,similarity_matrix,placeName_list,city_name):
        # similarity_matrix's size = (place number , place number)

        workbook = xlsxwriter.Workbook("Output/Doc2vec_output"".xlsx")
        worksheet_output = workbook.add_worksheet(city_name)
        
        for i in range(0,len(placeName_list[0])): #place number
            worksheet_output.write(i+1,0,placeName_list[0][i])
            worksheet_output.write(0,i+1,placeName_list[0][i])
            for j in range(0,len(placeName_list[0])):
                worksheet_output.write(i+1,j+1,similarity_matrix[i][j])

    
        workbook.close()
        