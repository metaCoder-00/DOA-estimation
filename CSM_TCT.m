function [theta, spectrum] = CSM_TCT()
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
    
    freqSnapshots = 100;
    nFFT = 64;
    
    
    c = 3e8;
    Ts = 1/fs;
    snapshots = freqSnapshots*nFFT;
    Ns = Ts*(0: snapshots - 1);
    margin = (c/f_end)/2;
    distance = margin*(0: sensorNum - 1)';
    
    receivedData = zeros(sensorNum, snapshots);
    manifoldMat = zeros(sensorNum, sourceNum);
    signalCovMat = [1, 0.99; 0.99, 1];
    for bandNum = 1: narrowBandNum
        f = f_begin + (bandNum - 1)*narrowBandwidth;
        signalAmp = mvnrnd(zeros(sourceNum, 1), signalCovMat, snapshots).';
        signalMat = [exp(-1j*2*pi*f*Ns); exp(-1j*2*pi*f*Ns)].*signalAmp;
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
    estManifold = zeros(sensorNum, sourceNum);
    focusFreqSignalCorreMat = 0;
    singularValsSum = 0;
    %------form signal correlation matrices and manifold in each frequency bin------%
    for freqPos = 1: nFFT
        if freqPos <= nFFT/2
            f = fs + (freqPos - 1)*fs/nFFT;
        else
            f = (f_end + bandwidth) - (freqPos - 1)*fs/nFFT;
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
%     focusFreq = (f_begin + f_end)/2;
    focusManifold = zeros(sensorNum, sourceNum);
    cost = inf;
    for freqPos = 1: nFFT/2
        f = fs + (freqPos - 1)*fs/nFFT;
        for col = 1: sourceNum
            focusManifold(:, col) = exp(-1j*2*pi*f*((distance*sind(preEstTheta(col)))/c));
        end
        focusData = focusManifold*focusFreqSignalCorreMat*focusManifold';
        thisCost = sum(abs(svd(focusData) - singularValsSum/nFFT).^2);
        if thisCost < cost
            focusFreq = f;
            cost = thisCost;
        end
    end
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
end