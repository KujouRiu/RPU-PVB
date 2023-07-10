import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def xlsx_to_csv_pd():
    data_xls = pd.read_excel(r"D:\Study\Code\RobustDet-master\1.xlsx", index_col=0)
    data_xls.to_csv('1.csv', encoding='utf-8')


if __name__ == '__main__':
    xlsx_to_csv_pd()
    data = pd.read_csv(r"D:\Study\Code\RobustDet-master\1.csv")

    data = np.array(data)