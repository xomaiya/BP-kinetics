clear all
close all
clc



Bbpmin=0.06;
Bbpmax=0.1;
Bpb=0.19;



%%
% List of cases

% % 1) Bpb=0.19, Bbp=0.02-0.031
% Bbpmin=0.02;
% Bbpmax=0.031;
% Bpb=0.19;
% Bbpmin=0.031;
% Bbpmax=0.02;
% Bpb=0.19;

% % 2) Bpb=0.4, Bbp=0.125-0.135
% Bbpmin=0.125;
% Bbpmax=0.135;
% Bpb=0.4;
% Bbpmin=0.135;
% Bbpmax=0.125;
% Bpb=0.4;

% % 3) Bbp=0.035-0.042, Bpb=0.56
% Bbpmin=0.035;
% Bbpmax=0.042;
% Bpb=0.56;
% Bbpmin=0.042;
% Bbpmax=0.035;
% Bpb=0.56;

% % 5) Bpb=0.5, Bbp=0.025-0.045
% Bbpmin=0.025;
% Bbpmax=0.045;
% Bpb=0.5;
% Bbpmin=0.045;
% Bbpmax=0.025;
% Bpb=0.5;

%%

% Bbpmin=0.125;
% Bbpmax=0.1;
% Bpb=0.5;

% Bpb=0.5, Bbp 0.125-0.1
B=[Bbpmin,Bbpmax,Bpb];



ve0=0;
ue0=0;
se0=0;
vi0=0;
ui0=0;
si0=0;

% ve0=0.9040;
% ue0=1.3204;
% se0=0.8372;
% vi0=0.0208;
% ui0=-0.1367;
% si0=0.2420;

f0=[ve0;ue0;se0;vi0;ui0;si0;];
[t,ve,ue,se,vi,ui,si]=FHN_syn_01_08_16w(f0,B);
% 
% 
% figure
% subplot(1,1,1,'FontSize',18);
% plot(t,ve)
% xlabel('t')
% ylabel('Ve')
 
% figure
% subplot(1,1,1,'FontSize',18);

L=length(t);
ti=linspace(t(1),t(end),2^round(log2(L)));
vei=interp1(t,ve,ti);
vea=sAnalytic(vei);
% plot3(ti,real(vea),imag(vea))
% xlabel('t')


figure
iptsetpref('ImshowBorder','tight');
subplot(3,1,1,'FontSize',22);
Tmax=2000;
grate=(Bbpmax-Bbpmin)/Tmax;
k=(Bbpmin+grate*t);

[ax,h1,h2]=plotyy(t,ve,t,k);
set(ax(1),'YTick',[-2:2:2])
set(ax(2),'FontSize',22,'Ylim',[0.06 0.1],'YTick',[0.06:0.02:0.1])


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
% plot((Bbpmin+grate*ti),abs(w(45,:)),'LineWidth',2)
% hold on
% plot((Bbpmin+grate*ti),abs(w(33,:)),'red','LineWidth',2)
% ylabel('|w(52.8,t)|, |w(38.4,t)|')
% xlabel('Bbp')

figure
subplot(1,1,1,'FontSize',22);
[amax,iamax]=max(abs(w(30:end,:)));
i1=find(iamax~=1);
plot((Bbpmin+grate*ti(i1)),2*a(iamax(i1))+2*a(29),'*')
ylabel('T')
xlabel('Bbp')
xlim([0.06 0.1])
% 
figure
subplot(1,1,1,'FontSize',22);
[amax,iamax]=max(abs(w(1:30,1:4000)));
i1=find(iamax~=1);
plot((Bbpmin+grate*ti(i1)),2*a(iamax(i1)),'redo')
hold on
[amax,iamax]=max(abs(w(30:end,3000:end)));
i1=find(iamax~=1);
plot((Bbpmin+grate*(ti(i1)+ti(4000))),2*a(iamax(i1))+2*a(29),'redo')
ylabel('T')
xlabel('Bbp')
xlim([0.06 0.1])
