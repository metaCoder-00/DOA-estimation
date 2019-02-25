function [theta, spectrum] = CSM_CSS()
    SNR = 15;
    sensorNum = 8;
    theta_S = [-15; 10];
    sourceNum = length(theta_S);
    %----Signal bandwidth: 2MHz, center freq: 11MHz fs: 10e6-----%
    fs = 10e6;
    f_begin = 10e6;
    f_end = 12e6;
    bandwidth = f_end - f_begin;
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
    
    preEstTheta = theta_S + rand();
    %---------form tansform matrix-------%
    transMat = zeros(sensorNum, sensorNum, nFFT);
    preManifoldMat = zeros(sensorNum, sourceNum);
    for col = 1: sourceNum
        manifoldMat(:, col) = exp(-1j*2*pi*f_end*((distance*sind(preEstTheta(col)))/c));
    end
    for freqPos = 1: nFFT
        f = (freqPos - 1)*(bandwidth/nFFT) + f_begin;
        for col = 1: sourceNum
            preManifoldMat(:, col) = exp(-1j*2*pi*f*((distance*sind(preEstTheta(col)))/c));
        end
        tempMatB = [zeros(sourceNum, sensorNum - sourceNum); eye(sensorNum - sourceNum)];
        transMat(:, :, freqPos) = [manifoldMat, tempMatB]*pinv([preManifoldMat, tempMatB]);
    end
    
    %--------form transformed covarince matrix-------%
    transCovMat = zeros(sensorNum, sensorNum);
    for freqPos = 1: nFFT
        data = dataSet(:, :, freqPos);
        covMat = (data*data')/size(data, 2);
        transCovMat = transCovMat + transMat(:, :, freqPos)*covMat*transMat(:, :, freqPos)';
    end
    transCovMat = transCovMat/nFFT;
    
    %-------MUSIC---------%
    [theta, spectrum] = MUSIC(transCovMat, f_end, sourceNum, sensorNum, margin);
end