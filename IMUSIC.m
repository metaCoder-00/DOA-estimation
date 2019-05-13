SNR = 15;
sensorNum = 8;
theta_S = [-15, 5];
sourceNum = length(theta_S);

f_begin = [600, 1000];
f_end = [1400, 1400];
fc = 1000;
bandwidth = f_end - f_begin;
fs = 2*max(bandwidth);
narrowBandwidth = 1;
narrowBandNum = bandwidth/narrowBandwidth;

freqSnapshots = 100;
nFFT = 256;
snapshots = freqSnapshots*nFFT;
Ts = (1/fs)*(0: snapshots - 1)' + 0.005;

c = 3e8;
margin = (c/max(f_end))/2;

test = 0;

receivedData = zeros(sensorNum, snapshots);
for n = 1: sourceNum
    for bandNum = 1: narrowBandNum(n)
        f = f_begin(n) + (bandNum - 1)*narrowBandwidth;
        steerVec = exp(-1j*2*pi*f*((margin*(0: sensorNum - 1)'*sind(theta_S(n))/c)));
        signalVec = exp(1j*2*pi*(f*Ts' + rand()));
        test = test + signalVec;
        receivedData = receivedData + steerVec*signalVec;
    end
end

receivedData = awgn(receivedData, SNR, 'measured');
receivedData = receivedData.*exp(-1j*2*pi*fc*Ts');
test = test.*exp(-1j*2*pi*fc*Ts');

dataSet = zeros(sensorNum, freqSnapshots, nFFT);
for slice = 1: freqSnapshots
    dataSlice = receivedData(:, (slice - 1)*nFFT + 1: slice*nFFT);
    dataSlice = fft(dataSlice, nFFT, 2);
    dataSet(:, slice, :) = dataSlice;
end

theta = (-30: 0.1: 30)';
spectrum = zeros(size(theta));
for freqBin = 1: nFFT
    if freqBin <= nFFT/2
        f = fc + (freqBin - 1)*fs/nFFT;
    else
        f = (fc - fs) + (freqBin - 1)*fs/nFFT;
    end
    data = dataSet(:, :, freqBin);
    covMat = data*data'/freqSnapshots;
    [eigVecs, eigVals] = eig(covMat);
    eigVals = diag(eigVals);
    [eigVals, index] = sort(eigVals);
    sourceNum = AIC(freqSnapshots, sensorNum, flip(eigVals));
    noiseSubspace = eigVecs(:, index(1: sensorNum - sourceNum));
    for n = 1: length(theta)
        steerVec = exp(-1j*2*pi*f*(margin*(0: sensorNum - 1)'*sind(theta(n)))/c);
        spectrum(n) = spectrum(n) + (steerVec'*steerVec)./...
                        (steerVec'*(noiseSubspace*noiseSubspace')*steerVec);
    end
end
spectrum = spectrum./nFFT;

plot(theta, 10*log10(abs(spectrum)/max(abs(spectrum))))
grid on
hold on
for n = 1: length(theta_S)
    plot([theta_S(n), theta_S(n)], get(gca, 'YLim'), '--r')
end
hold off
set(gca, 'XTICK', -30: 5: 30)
xlabel('angle/degree')
ylabel('spectrum/dB')
title('IMUSIC')