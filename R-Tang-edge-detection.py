import statistics
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

low_limit,up_limit,M,N=0.4,0.85,20,5

#input
os.chdir('D:\sai kumar\Project\Shankar')
A=pd.read_excel(io='ab.xlsx',sheet_name='Sheet1')
A.fillna(0, inplace=True)
A=A.loc[A.LST !='<Null>']
A=A.loc[A.LST != 0]


def RMSE(y,y_lr):
    sm=0
    for i in range(len(y)):
        sm=sm+(y[i]-y_lr[i])**2
    op=(sm/len(y))**0.5
    return op

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

# function used to seprate the far data
def fun_max(ip):
    l=len(ip)
    nval=statistics.mean(ip)-statistics.pstdev(ip)
    if nval<0:
        nval=nval*(-1)
    op=[]
    for i in ip:
        if i >= nval:
            op.append(i)
    if l==len(op):
        return op
    else:
        return fun_max(op)
def fun_min(ip):
    l=len(ip)
    pval=statistics.mean(ip)+statistics.pstdev(ip)
    op=[]
    for i in ip:
        if i <= pval:
            op.append(i)
    if l==len(op):
        return op
    else:
        return fun_min(op)
    
# function applied to 2 lists for removing zero cordinates
def funlst(x,y):
    x1,y1=[],[]
    for i in range(len(x)):
        if y[i] !=0:
            y1.append(y[i])
            x1.append(x[i])
    return x1,y1

# function applied to 2 lists for liner regrassion logic
def  funlr(x,y,error):
    x,y=funlst(x,y)
    x1=numpy.array(x).reshape(-1,1)
    y1=numpy.array(y).reshape(-1,1)
    LR=LinearRegression()
    LR.fit(x1,y1)
    y1=LR.predict(x1)
    y_lr=[y1[i][0] for i in range(len(y1))]
    rmse=error*RMSE(y,y_lr)
    op,cnt=[],0
    for i in range(len(y)):
        dis=y[i]-y_lr[i]
        if dis<0:
            dis=dis*(-1)
        if dis <= rmse:
            op.append(y[i])
        else:
            op.append(0)
            cnt=cnt+1
    if sum(op)==0 or len(x)==2:
        return 'increase error value'
    elif cnt != 0:
        return funlr(x,op,error)  
    else:
        m=(y_lr[0]-y_lr[1])/(x[0]-x[1])
        c=y_lr[0]-(m*x[0])
        rmse=rmse/error
        y1=numpy.array(y).reshape(-1,1)
        r2=LR.score(x1,y1)
        return x,op,y_lr,m,c,rmse,r2
# function is used to find FOS
def fun(x,y):
    FOS=2
    for i in range(10000):
        if len(funlr(x,y,FOS))==7:
            return FOS,funlr(x,y,FOS)
            break
        else:
            FOS=FOS+0.5
# Main Algorithem 
y_max,y_min=[],[]
sup_set=rang(low_limit,up_limit,M)
for i in range(M):
    ll=sup_set[i]
    ul=sup_set[i+1]
    sub_set=rang(ll,ul,N)
    iop_max,iop_min=[],[]
    for j in range(N):
        lr=sub_set[j]
        ur=sub_set[j+1]
        filt=A.loc[(A['NDVI'] >= lr) & (A['NDVI'] < ur)]
        if len(filt)==0:
            iop_max.append(0)
            iop_min.append(0)
        else:
            iop_max.append(max(filt.LST))
            iop_min.append(min(filt.LST))
    iop_max=list(filter(lambda a: a != 0, iop_max))
    iop_min=list(filter(lambda a: a != 0, iop_min))
    if len(iop_max)==0:
        y_max.append(0)
    else:
        y_max.append(statistics.mean(fun_max(iop_max)))
    if len(iop_min)==0:
        y_min.append(0)
    else:
        y_min.append(statistics.mean(fun_min(iop_min)))
x=rang(sup_set[0]+(sup_set[1]-sup_set[0])/2,sup_set[M]-(sup_set[1]-sup_set[0])/2,M-1)

# collecting the data required for liner regrassion curve  print(fun(x,y_max))
dry,wet=fun(x,y_max),fun(x,y_min)
c_dry=dry[1][4]
m_dry=dry[1][3]
r_dry=dry[1][5]
r2_dry=dry[1][6]
foc_dry=dry[0]
yip_dry=dry[1][1]
xip_dry=dry[1][0]
c_wet=wet[1][4]
m_wet=wet[1][3]
r_wet=wet[1][5]
r2_wet=wet[1][6]
foc_wet=wet[0]
yip_wet=wet[1][1]
xip_wet=wet[1][0]
yop_dry=[(i*m_dry)+c_dry for i in x]
yop_wet=[(i*m_wet)+c_wet for i in x]
plt.scatter(A.NDVI,A.LST,color='green')
plt.plot(x,yop_dry,color='red',label='Dry edge')
plt.scatter(xip_dry,yip_dry,color='red')
plt.plot(x,yop_wet,color='blue',label='Wet edge')
plt.scatter(xip_wet,yip_wet,color='blue')
plt.legend()
plt.xlabel('NDVI')
plt.ylabel('LST')
plt.show()
print(f'Dry edge :- m={m_dry} c={c_dry} R_Square={r2_dry} Rmse={r_dry} FOS={foc_dry}')
print(f'Wet edge :- m={m_wet} c={c_wet} R_Square={r2_wet} Rmse={r_wet} FOS={foc_wet}')
