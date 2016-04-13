import numpy as np
import pandas as pd

file_object = open('horse-colic.data')
try:
    all_text_lines = file_object.readlines( )
    all_text = []
    for line in all_text_lines:
    	all_text.append(line.strip().split(" "))

    l = [3,4,5,15,18,19,21]

    for line in all_text:
    	for i in range(0, len(line)):
            if line[i] != '?' and i in l:
    			line[i] = float(line[i])

    df = pd.DataFrame(all_text)
    df = df.replace("?", np.nan)
    for i in range(0,len(all_text[0])-1):
        df[i] = df[i].fillna(df[i].mode().values[0])

    for c in l:
        df[c].astype(np.float64)
    df.to_excel("out.xls")

    for c in l :
        df[c] = (df[c] - df[c].min())/(df[c].max() - df[c].min())
    df.to_excel("out2.xls")
finally:
    file_object.close( )
