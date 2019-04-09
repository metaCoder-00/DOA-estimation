function [theta, spectrum] = CSM_CSS()
    SNR = 15;
    sensorNum = 8;
    theta_S = [-10; 0];
    sourceNum = length(theta_S);
    %----Signal bandwidth: 4MHz, center freq: 10MHz fs: 8e6-----%
    f_begin = 8e6;
    f_end = 12e6;
    bandwidth = f_end - f_begin;
    fs = 2*bandwidth;
    narrowBandwidth = 1e2;
    narrowBandNum = bandwidth/narrowBandwidth;
    
    freqSnapshots = 200;
    nFFT = 64;
    
    
    c = 3e8;
    Ts = 1/fs;
    snapshots = freqSnapshots*nFFT;
    Ns = Ts*(0: snapshots - 1);
    margin = (c/f_end)/2;
    distance = margin*(0: sensorNum - 1)';
    
    receivedData = zeros(sensorNum, snapshots);
    manifoldMat = zeros(sensorNum, sourceNum);
    signalCovMat = [1, 0; 0, 1];
    for bandNum = 1: narrowBandNum
        f = f_begin + (bandNum - 1)*narrowBandwidth;
        signalAmp = mvnrnd(zeros(sourceNum, 1), signalCovMat, snapshots).';
        signalMat = [exp(1j*2*pi*f*Ns); exp(1j*2*pi*f*Ns)].*signalAmp;
        for col = 1: sourceNum
            manifoldMat(:, col) = exp(-1j*2*pi*f*((distance*sind(theta_S(col)))/c));
        end
        receivedData = receivedData + manifoldMat*signalMat;
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
    %---------form tansform matrix-------%
    focusFreq = (f_begin + f_end)/2;
%     auxiAngles = [-25; -20; -15; 10; 15; 20];
%     focusAuxiManifoldMat = zeros(sensorNum, length(auxiAngles));
%     freqBinManifoldMat = zeros(sensorNum, length(auxiAngles));
    transMat = zeros(sensorNum, sensorNum, nFFT);
    preManifoldMat = zeros(sensorNum, sourceNum);
    noiseCovMat = zeros(sensorNum, sensorNum);
    for col = 1: sourceNum
        manifoldMat(:, col) = exp(-1j*2*pi*focusFreq*((distance*sind(preEstTheta(col)))/c));
    end
%     for col = 1: length(auxiAngles)
%             focusAuxiManifoldMat(:, col) = exp(-1j*2*pi*focusFreq*((distance*sind(auxiAngles(col)))/c));
%     end
    for freqPos = 1: nFFT
        if freqPos <= nFFT/2
            f = fs + (freqPos - 1)*fs/nFFT;
        else
            f = (f_end + bandwidth) - (freqPos - 1)*fs/nFFT;
        end
        for col = 1: sourceNum
            preManifoldMat(:, col) = exp(-1j*2*pi*f*((distance*sind(preEstTheta(col)))/c));
        end
        tempMatB = [zeros(sourceNum, sensorNum - sourceNum); eye(sensorNum - sourceNum)];
        transMat(:, :, freqPos) = [manifoldMat, tempMatB]*pinv([preManifoldMat, tempMatB]);
%         for col = 1: length(auxiAngles)
%             freqBinManifoldMat(:, col) = exp(-1j*2*pi*f*((distance*sind(auxiAngles(col)))/c));
%         end
%         transMat(:, :, freqPos) = [manifoldMat, focusAuxiManifoldMat]*pinv([preManifoldMat, freqBinManifoldMat]);
        noiseCovMat = noiseCovMat + transMat(:, :, freqPos)*transMat(:, :, freqPos)';
    end
    
    %--------prewhite noise covariance matrix--------%
    upperMat = chol(noiseCovMat);
    
    %--------form transformed covarince matrix-------%
    transCovMat = zeros(sensorNum, sensorNum);
    for freqPos = 1: nFFT
        data = dataSet(:, :, freqPos);
        covMat = (data*data')/size(data, 2);
        transCovMat = transCovMat + transMat(:, :, freqPos)*covMat*transMat(:, :, freqPos)';
    end
    transCovMat = pinv(upperMat')*transCovMat*pinv(upperMat);
    
    %-------MUSIC---------%
    [theta, spectrum] = MUSIC(transCovMat, focusFreq, sourceNum, sensorNum, margin);
end