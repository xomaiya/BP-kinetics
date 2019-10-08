function w=fftMorlet(t,fp,a,omega0);


N=length(t);
%Fourier transform 
F=fft(fp);
nrm=2*pi/(t(end)-t(1));
omega_=([(0:(N/2)) (((-N/2)+1):-1)])*nrm;
%Convolution
if a(1)==0
    w(1,:)=fp*exp(-omega0^2/2);
    k1=2;
else
    k1=1;
end
for k=k1:length(a);
    omega_s=a(k)*omega_;
    window=exp(-(omega_s-omega0).^2/2);
    cnv(k,:)=window.*F;
    w(k,:)=ifft(cnv(k,:));
end

