SNR = 15;
sensorNum = 8;
theta_S = [-15, 5];
sourceNum = length(theta_S);

f_begin = [800e6, 900e6];
f_end = [1200e6, 1500e6];
fc = 1000e6;
bandwidth = f_end - f_begin;
fs = 2*max(bandwidth);

freqSnapshots = 100;
nFFT = 128;
snapshots = freqSnapshots*nFFT;
Ts = (1/fs)*(0: snapshots - 1)' + 0.005;

c = 3e8;
margin = (c/max(f_end))/2;

receivedData = zeros(sensorNum, snapshots);
for m = 1: sensorNum
    for n = 1: sourceNum
        delay = (m - 1)*margin*sind(theta_S(n))/c;
        k = bandwidth(n)/Ts(end);
%         signal = exp(1j*2*pi*(f_begin(n)*Ts + (1/2)*k*Ts.^2));
        receivedData(m, :) = receivedData(m, :) + exp(1j*2*pi*(f_begin(n)*(Ts - delay) + ...
                                (1/2)*k*(Ts - delay).^2)).';
    end
end
receivedData = awgn(receivedData, SNR, 'measured');
receivedData = receivedData.*exp(-1j*2*pi*fc*Ts).';

dataSet = zeros(sensorNum, freqSnapshots, nFFT);
for slice = 1: freqSnapshots
    dataSlice = receivedData(:, (slice - 1)*nFFT + 1: slice*nFFT);
    dataSlice = fft(dataSlice, nFFT, 2);
    dataSet(:, slice, :) = dataSlice;
end

theta = (-30: 0.1: 30)';
spectrum = zeros(length(theta), 1);
for freqBin = 1: nFFT
    data = dataSet(:, :, freqBin);
    covMat = data*data'/freqSnapshots;
    [eigVecs, eigVals] = eig(covMat);
    eigVals = diag(eigVals);
    [~, index] = sort(eigVals);
    noiseSubspace = eigVecs(:, index(1: sensorNum - sourceNum));
    if freqBin <= nFFT/2
        f = fc + (freqBin - 1)*fs/nFFT;
    else
        f = (fc - fs) + (freqBin - 1)*fs/nFFT;
    end
    for n = 1: length(theta)
        steerVec = exp(-1j*2*pi*f*(margin*(0: sensorNum - 1)'*sind(theta(n)))/c);
        spectrum(n) = spectrum(n) + ...
                        (steerVec'*steerVec)/(steerVec'*(noiseSubspace*noiseSubspace')*steerVec);
    end
end
spectrum = spectrum/nFFT;

plot(theta, 10*log10(abs(spectrum)/max(abs(spectrum))))
grid on
hold on
for n = 1: sourceNum
    plot([theta_S(n), theta_S(n)], get(gca, 'YLim'), '--r')
end
hold off
set(gca, 'XTICK', -30: 5: 30)
xlabel('angle/degree')
ylabel('spectrum/dB')
title('TCT-LFM')