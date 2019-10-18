clear all
close all
clc

% Bpbmin=0.2;
% Bpbmax=0.6;
% Bbp=0.021;

% Bpbmin=0.22;
% Bpbmax=0.26;
% Bbp=0.12;

Bpbmin=0.15;
Bpbmax=0.25;
Bbp=0.08;

%%
% List of cases

% % 4) Bbp=0.04, Bpb=0.53-0.6
% Bpbmin=0.53;
% Bpbmax=0.6;
% Bbp=0.04;
% Bpbmin=0.6;
% Bpbmax=0.53;
% Bbp=0.04;

% % 6) Bbp=0.03, Bpb=0.7-0.85
% Bpbmin=0.7;
% Bpbmax=0.85;
% Bbp=0.03;
% Bpbmin=0.85;
% Bpbmax=0.7;
% Bbp=0.03;

% 7) Bbp=0.06, Bpb=0.008-0.06
% Bpbmin=0.008;
% Bpbmax=0.06;
% Bbp=0.06;
% Bpbmin=0.06;
% Bpbmax=0.008;
% Bbp=0.06;

%%



B=[Bpbmin,Bpbmax,Bbp];


ve0=0;
ue0=0;
se0=0;
vi0=0;
ui0=0;
si0=0;

f0=[ve0;ue0;se0;vi0;ui0;si0;];
[t,ve,ue,se,vi,ui,si]=FHN_syn_01_08_16w2(f0,B);


% figure
% subplot(1,1,1,'FontSize',18);
% plot(t,ve)
% xlabel('t')
% ylabel('Ve')
% 
% figure
% subplot(1,1,1,'FontSize',22);
L=length(t);
ti=linspace(t(1),t(end),2^round(log2(L)));
vei=interp1(t,ve,ti);
vea=sAnalytic(vei);
% plot3(ti,real(vea),imag(vea))
% xlabel('t')



figure
iptsetpref('ImshowBorder','tight');
subplot(3,1,1,'FontSize',22);

grate=(Bpbmax-Bpbmin)/2000;
k=Bpbmin+grate*t;

[ax,h1,h2]=plotyy(t,ve,t,k);
set(ax(1),'YTick',[-2:2:2])
% set(ax(2),'FontSize',22,'Ylim',[0.22 0.26],'YTick',[0.22:0.02:0.26])
set(ax(2),'FontSize',22,'Ylim',[0.45 0.46],'YTick',[0.45:0.01:0.46])

xlabel('t')
ylabel('\nu_p')
ylim([-2.2 2.2])


omega0=pi;
a=linspace(0,60,101);
T=a*2*pi/omega0;
w=fftMorlet(ti,vea,a,omega0);
subplot(3,1,[2 3],'FontSize',22);
pcolor(ti,T,abs(w));
shading flat
xlabel('t')
ylabel('T')
title('|w|')

% figure
% subplot(1,1,1,'FontSize',22);
% L=length(t);
% plot3(ti,real(vea),imag(vea))
% xlabel('t')
% ylabel('\nu_p')
% zlabel('H[\nu_p]')

iptsetpref('ImshowBorder','tight');

%% Random IC

n=100;

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
    