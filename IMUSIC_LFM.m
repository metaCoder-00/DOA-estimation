SNR = 15;
sensorNum = 8;
theta_S = [-20, 15];
sourceNum = length(theta_S);

f_begin = [800e6, 900e6];
f_end = [1200e6, 1100e6];
bandwidth = f_end - f_begin;
fs = 2*(max(f_end) - max(f_begin));

freqSnapshots = 100;
nFFT = 64;
snapshots = freqSnapshots*nFFT;
Ts = (1/fs)*(0: snapshots - 1)';

c = 3e8;
margin = (c/max(f_end))/2;
distance = margin*(0: sensorNum - 1)';

receivedData = zeros(sensorNum, snapshots);
for arrayNum = 1: sensorNum
    for signalNum = 1: sourceNum
        delay = (margin*sind(theta_S(signalNum)))/c;
        grad = bandwidth(signalNum)/Ts(end);
        receivedData(arrayNum, :) = receivedData(arrayNum, :) + (randn(snapshots, 1).*...
                                    exp(1j*2*pi*(f_begin(signalNum)*(Ts - delay) + ...
                                    (1/2)*grad*(Ts - delay).^2))).';                                    
    end
end
receivedData = awgn(receivedData, SNR, 'measured');

dataSet = zeros(sensorNum, freqSnapshots, nFFT);
for slice = 1: freqSnapshots
    dataSlice = receivedData(:, (slice - 1)*nFFT + 1: slice*nFFT);
    for eachSensor = 1: sensorNum
        dataSet(eachSensor, slice, :) = fft(dataSlice(eachSensor, :), nFFT);
    end
end

preEstTheta = theta_S + 0.25*randn();
    estManifold = zeros(sensorNum, sourceNum);
    focusFreqSignalCorreMat = 0;
    singularValsSum = 0;
    %------form signal correlation matrices and manifold in each frequency bin------%
    for freqPos = 1: nFFT
        if freqPos <= nFFT/2
            f = fs + (freqPos - 1)*fs/nFFT;
        else
            f = (max(f_end) + fs/2) - (freqPos - 1)*fs/nFFT;
        end
        data = dataSet(:, :, freqPos);
        covMat = (data*data')/size(data, 2);
        [~, eigenVals] = eig(covMat);
        eigenVals = diag(eigenVals);
        noisePower = min(eigenVals);
        cleanedData = covMat - noisePower*eye(sensorNum);
        singularValsSum = singularValsSum + svd(cleanedData);
        for col = 1: sourceNum
            estManifold(:, col) = exp(-1j*2*pi*f*((distance*sind(preEstTheta(col)))/c));
        end
        signalCorreMat = pinv(estManifold'*estManifold)*estManifold'*cleanedData*estManifold*pinv(estManifold'*estManifold);
        focusFreqSignalCorreMat = focusFreqSignalCorreMat + signalCorreMat;
    end
    focusFreqSignalCorreMat = focusFreqSignalCorreMat/nFFT;
    %------form focus matrices------%
    focusFreq = (max(f_begin) + max(f_end))/2;
    focusManifold = zeros(sensorNum, sourceNum);
%     cost = inf;
%     for freqPos = 1: nFFT/2
%         f = fs + (freqPos - 1)*fs/nFFT;
%         for col = 1: sourceNum
%             focusManifold(:, col) = exp(-1j*2*pi*f*((distance*sind(preEstTheta(col)))/c));
%         end
%         focusData = focusManifold*focusFreqSignalCorreMat*focusManifold';
%         thisCost = sum(abs(svd(focusData) - singularValsSum/nFFT).^2);
%         if thisCost < cost
%             focusFreq = f;
%             cost = thisCost;
%         end
%     end
    for col = 1: sourceNum
        focusManifold(:, col) = exp(-1j*2*pi*focusFreq*((distance*sind(preEstTheta(col)))/c));
    end
    focusData = focusManifold*focusFreqSignalCorreMat*focusManifold';
    [focusEigSubspace, eigenVals] = eig(focusData);
    eigenVals = diag(eigenVals);
    [~, index] = sort(eigenVals, 'descend');
    focusEigSubspace = focusEigSubspace(:, index);
    
    covMat_TCT = 0;
    for freqPos = 1: nFFT
        data = dataSet(:, :, freqPos);
        covMat = (data*data')/size(data, 2);
        [~, eigenVals] = eig(covMat);
        eigenVals = diag(eigenVals);
        noisePower = min(eigenVals);
        cleanedData = covMat - noisePower*eye(sensorNum);
        [eigSubspace, eigenVals] = eig(cleanedData);
        eigenVals = diag(eigenVals);
        [~, index] = sort(eigenVals, 'descend');
        eigSubspace = eigSubspace(:, index);
        transMat = focusEigSubspace*eigSubspace';
        covMat_TCT = covMat_TCT + transMat*cleanedData*transMat';
    end
    covMat_TCT = covMat_TCT/nFFT;
    
    %-------MUSIC---------%
    [theta, spectrum] = MUSIC(covMat_TCT, focusFreq, sourceNum, sensorNum, margin);

plot(theta, 10*log10(abs(spectrum)/max(abs(spectrum))))
grid on
set(gca, 'XTICK', -30: 5: 30)
xlabel('angle/degree')
ylabel('spectrum/dB')
title('TCT-LFM')