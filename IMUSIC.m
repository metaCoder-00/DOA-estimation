function [theta, spectrum] = IMUSIC()
    SNR = 15;
    sensorNum = 8;
    theta_S = [-20; 10];
    sourceNum = length(theta_S);
%----Signal bandwidth: 4MHz, center freq: 10MHz fs: 10MHz-----%
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
    for bandNum = 1: narrowBandNum
        f = f_begin + (bandNum - 1)*narrowBandwidth;
        signalMat = [exp(1j*2*pi*f*Ns); exp(1j*2*pi*f*Ns)];
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
    spectrum = zeros(size(theta, 1), nFFT);
    for freqPos = 1: nFFT
        if freqPos <= nFFT/2
            f = fs + (freqPos - 1)*fs/nFFT;
        else
            f = (f_end + bandwidth) - (freqPos - 1)*fs/nFFT;
        end
        [~, narrowBandSpec] = MUSIC(dataSet(:, :, freqPos), f, sourceNum, sensorNum, margin);
%         spectrum = spectrum + narrowBandSpec;
        spectrum(:, freqPos) = narrowBandSpec;
    end
%     spectrum = spectrum/nFFT;
    
end