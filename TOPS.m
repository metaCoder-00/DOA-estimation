function [theta, spectrum] = TOPS()
    SNR = 15;
    sensorNum = 8;
    theta_S = [-20; 13];
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
    for bandNum = 1: narrowBandNum
        f = f_begin + (bandNum - 1)*narrowBandwidth;
        signalMat = [exp(-1j*2*pi*f*Ns); exp(-1j*2*pi*f*Ns)];
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
    
    theta = (-30: 0.1: 30)';
    spectrum = zeros(size(theta));
    data = dataSet(:, :, 1);
    covMat = (data*data')/size(data, 2);
    [eigenVec, eigenVals] = eig(covMat);   % Eigen factorization
    eigenVals = diag(eigenVals);
    [~, index] = sort(eigenVals, 'descend');
    signalSubspace = eigenVec(:, index(1: sourceNum));
    for itr = 1: length(theta)        
        estimator = [];
        for freqPos = 2: nFFT
            data = dataSet(:, :, freqPos);
            covMat = (data*data')/size(data, 2);
            [eigenVec, eigenVals] = eig(covMat);   % Eigen factorization
            eigenVals = diag(eigenVals);
            [~, index] = sort(eigenVals, 'ascend');
            noiseSubspace = eigenVec(:, index(1: sensorNum - sourceNum));
            
            f = (freqPos - 1)*(bandwidth/nFFT) + f_begin;
            deltaF = f - f_begin;
            steeringVec = exp(-1j*2*pi*deltaF*(margin*(0: sensorNum - 1)'*sind(theta(itr)))/c);
            unitaryMat = diag(steeringVec);
            projectionMat = eye(sensorNum) - pinv(steeringVec'*steeringVec)*(steeringVec*steeringVec');
            tempMatU = projectionMat*unitaryMat*signalSubspace;
            estimator = [estimator, tempMatU'*noiseSubspace];
        end
        singularVals = svd(estimator);
        spectrum(itr) = 1/(singularVals(end));
        
    end
    
end