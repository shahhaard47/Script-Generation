import pandas as pd
import numpy as np
from itertools import chain,repeat

f = open("test3_preds.txt",errors='ignore')
f2 = open("test3.txt",errors='ignore')
l2 = f2.readlines()
l2 = [x.strip() for x in l2]

l = f.readlines()
l = [x.strip() for x in l]

f2 = open("test3.txt")

cols = ["Seed Text",'<Comedy>', '<Action>', '<Adventure>', '<Crime>', '<Drama>', '<Fantasy>', '<Horror>', '<Music>', '<Romance>', '<Sci-Fi>', '<Thriller>']
df = {}
df["Seed Text"] = l2
l3 = np.array(l).reshape(20,11).T
for i in range(1,len(cols)):
    c = cols[i]
    l4 = [x.replace(c,'') for x in l3[i-1]]
    df[c] = l4
df = pd.DataFrame(df)
df2 = []
for i in range(len(df)):
    for c in cols[1:]:
        df2.append((c,df.loc[i,"Seed Text"],df.loc[i,c]))
df2 = pd.DataFrame(df2,columns=["Genre","Seed Text","Generated"])
df2.to_csv("output.csv",index=False)