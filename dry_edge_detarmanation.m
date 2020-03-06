clc
clear all
cd 'D:\sai kumar\Project\Shankar'
A=xlsread('LSTNDVI.xlsx');
M=20;
N=20;
MI=[0:(1-0)/M:1];
for m=1:M
    sm=MI(m);
    em=MI(m+1);
    NI=[sm:(em-sm)/N:em];
    for n=1:N
       plc=find(A(:,2)>=NI(n) & A(:,2)<NI(n+1));
       if size(plc,1)~=0
           op1(n)=max(A(plc,1));
       else
           op1(n)=0;
       end
    end
    op1=nonzeros(op1);
    for n=1:N
        if length(find(op1<mean(op1)-std(op1)))==0
            op(m)=mean(op1);
            break
        else
            op1=op1(op1>(mean(op1)-std(op1)));
        end
    end
end
Fri=(1/(2*M)):(1/M):((2*M)-1)/(2*M);
Fri=Fri.*(~isnan(op));
op(isnan(op))=0;
op=[nonzeros(op'),nonzeros(Fri')];

% Linear regrassion y=mx+c
x=op(:,2);
y=op(:,1);

for i=1:size(op,1)
    x1=[ones(length(x),1),x];
    output=(x1\y);
    res = y - x1*output;
    rsquare= 1 - var(res)/var(y);
    if sum(res<(rsquare*2)&res>-(rsquare*2))==size(x,1)
        m=output(2)
        c=output(1)
        break
    else
        x=x(res<(rsquare*2)&res>-(rsquare*2));
        y=y(res<(rsquare*2)&res>-(rsquare*2));
    end
end

% plotting graph
plot(x,y,'.b');
xlabel('NDVI')
ylabel('LST')
title(['LST = ' num2str(m) ' NDVI + ' num2str(c) '    R_2 = ' num2str(rsquare)])
hold on
ynew=m*x+c;
plot(x,ynew,'r')
hold off