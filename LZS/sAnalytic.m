function fa=sAnalytic(f);

N=length(f);
%Cut-off of negative frequencies
F=fft(f);
F=[2*F(1:N/2),zeros(1,N/2)];
%Output: the analytic signal
fa=ifft(F);

