import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
#input
low_limit,up_limit,M = 0.4,0.85,20

os.chdir('D:\sai kumar\Project\Shankar')
A=pd.read_excel(io='ab.xlsx',sheet_name='Sheet1')
A.fillna(0, inplace=True)
A=A.loc[A.LST !='<Null>']
A=A.loc[A.LST != 0]   
#function is used to divide the given intervel 
def rang(l1,l2,I):
    r=((l2-l1)/I)
    op=[l1]
    sm=l1
    for i in range(I-1):
        sm=float("{0:.4f}".format(sm+r))
        op.append(sm)
    op.append(l2)
    return op
# function applied to 2 lists for removing zero cordinates
def funlst(x,y):
    x1,y1=[],[]
    for i in range(len(x)):
        if y[i] !=0:
            y1.append(y[i])
            x1.append(x[i])
    return x1,y1
def funlr(x,y):
    x,y=funlst(x,y)
    x1=numpy.array(x).reshape(-1,1)
    y1=numpy.array(y).reshape(-1,1)
    LR=LinearRegression()
    LR.fit(x1,y1)
    rmse=LR.score(x1,y1)
    y1=LR.predict(x1)
    y_lr=[y1[i][0] for i in range(len(y1))]
    m=(y_lr[0]-y_lr[1])/(x[0]-x[1])
    c=y_lr[0]-(m*x[0])
    return x,y,y_lr,m,c,rmse

# Main Algorithem 
sup_set=rang(low_limit,up_limit,M)
y_max,y_min=[],[]
for i in range(M):
    ll=sup_set[i]
    ul=sup_set[i+1]
    filt=A.loc[(A['NDVI'] >= ll) & (A['NDVI'] < ul)]
    if len(filt)==0:
        y_max.append(0)
        y_min.append(0)
    else:
        y_max.append(max(filt.LST))
        y_min.append(min(filt.LST))

x=rang(sup_set[0]+(sup_set[1]-sup_set[0])/2,sup_set[M]-(sup_set[1]-sup_set[0])/2,M-1)
# collecting the data required for liner regrassion curve  
dry,wet=funlr(x,y_max),funlr(x,y_min)
plt.scatter(A.NDVI,A.LST,color='green')
plt.plot(dry[0],dry[2],color='red',label='Dry edge')
plt.scatter(dry[0],dry[1],color='red')
plt.plot(wet[0],wet[2],color='blue',label='Wet edge')
plt.scatter(wet[0],wet[1],color='blue')
plt.legend()
plt.xlabel('NDVI')
plt.ylabel('LST')
plt.show()
print(f'Dry edge :- m={dry[3]} c={dry[4]} R_Square={dry[5]}')
print(f'Wet edge :- m={wet[3]} c={wet[4]} R_Square={wet[5]}')