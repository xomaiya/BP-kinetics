function [t,ve,ue,se,vi,ui,si]=FHN_syn_01_08_16m(f0,B);

%FHN
ae=0.5;
be=0.8;
ai=0.5;
bi=0.8;
phie=0.3;
phii=0.3;
Ie=0.5;

Ii=0;

%synaptic coupling
A2=1;
Bbp=B(1);
A1=1;
Bpb=B(2);
vsl=0.1;
Eex=0;
Ein=-5;
%conductances for synaptic coupling

gei=0.5;%0.57;
gie=0.5;%0.1;


%initial conditions for ve
f01 = f0;

%%zeit

%Nt=1000;%number of time nodes
Tmax=2000;%maximal time of calculations
t = [0 Tmax];%linspace(0,Tmax,Nt);



%equations


%options = odeset('RelTol', 1e-8,'AbsTol',1e-9);
% [t,V] = ode15s(@f1,t,f01,options);
[t,V] = ode23s(@f1,t,f01);
ve=V(:,1);
ue=V(:,2);
se=V(:,3);
vi=V(:,4);
ui=V(:,5);
si=V(:,6);


  % -----------------------------------------------------------------------

function dydt = f1(t,y);
    dydt = [y(1)-y(1)^3/3-y(2)+Ie+gie*y(6)*(Ein-y(1));%+go*so2*(Ein-ve)+gie*si*(Ein-ve); 
            phie*(y(1)+ae-be*y(2));
            A1/2*(1+tanh(y(1)/vsl))*(1-y(3))-Bpb*y(3);
            
            y(4)-y(4)^3/3-y(5)+Ii+gei*y(3)*(Eex-y(4));
            phii*(y(4)+ai-bi*y(5));
            A2/2*(1+tanh(y(4)/vsl))*(1-y(6))-Bbp*y(6);];
end
  % ----------------------------------------------------------------------- 
end