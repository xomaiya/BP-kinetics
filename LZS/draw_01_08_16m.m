clear all
close all
clc

Bbp=0.08;
Bpb=0.19;
B=[Bbp,Bpb];


ve0=0;%1.601176; %L2-1.601176;
ue0=0;%0.636761;%L2-0.636761;  %unison-0.635973;
se0=0;%0.73242;%0.73242;%0.732364;
vi0=0;%-0.902724;%-1.277757;%-0.920928;
ui0=0;%-0.419631;%-0.053213;%-0.408815;
si0=0;%0.000233;%0.022778;%0.000228;

f0=[ve0;ue0;se0;vi0;ui0;si0;];
[t,ve,ue,se,vi,ui,si]=FHN_syn_01_08_16m(f0,B);


figure
subplot(1,1,1,'FontSize',18);
plot(t,ve)
% 
% T=periodf(t,ve,0)

%% Random IC

n=10;

for j=1:n
    j
    ve0=2*rand-1;
    ue0=2*rand-1;
    se0=rand;
    vi0=2*rand-1;
    ui0=2*rand-1;
    si0=rand;
    
    f0=[ve0;ue0;se0;vi0;ui0;si0;];
    [t,ve,ue,se,vi,ui,si]=FHN_syn_01_08_16m(f0,B);
%     maxve(j)=max(ve);
    T(j)=periodf(t,ve,0);
end

figure
subplot(1,1,1,'FontSize',18);
binT=[10:10:160];
hist(T,binT)
xlabel('Period')
ylabel('Fraction of attractors')
    