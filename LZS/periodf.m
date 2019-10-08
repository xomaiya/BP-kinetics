function T=periodf(t,u,co);

L=length(t);
t=t(round(0.25*L):L);
u=u(round(0.25*L):L);

j=2:length(t)-2;
[wmi,wm]=find((u(j-1)<u(j))&(u(j+1)<u(j))&(u(j)>co));
j=2:length(wmi);
T=mean(t(wmi(j))-t(wmi(j-1)));