import pandas as pd
import numpy as np

def flines(f,l):
    sz = len(l)
    for i in range(sz-1):
        f.write(l[i])
        f.write('\n')
    f.write(l[-1])

df = pd.read_csv("genre.csv")
f = open("train.source","w")
f2 = open("train.target","w")
f3 = open("val.source","w")
f4 = open("val.target","w")
for i in range(10):
    dl = str(bytes(' '.join(df.loc[i,"script"].split()),'utf-8').decode('ascii','ignore').encode("ascii"))[1:].replace(r'\'','\'').split()
    if(len(dl)<100):
        continue
    print(df.iloc[i,0])
    dl2 = []
    for i in range(0,len(dl),56):
        dl2.append(' '.join(dl[i:i+56]))
    lim = (len(dl2)*3)//4
    flines(f,dl2[:lim-1])
    flines(f2,dl2[1:lim])
    flines(f3,dl2[lim:-1])
    flines(f4,dl2[lim+1:])