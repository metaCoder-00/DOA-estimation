function [theta, spectrum] = MUSIC(data, f, sourceNum, sensorNum)
%---@Input: data (x = As + n)------%
%---@       f: frequency (Hz)---%
%---@       sourceNum: number of sources---%
%---@       sensorNum: number of sensors---%
%---@Output: spectrum of MUSIC--%
    covMat = (data*data')/size(data, 2);    % Covariance matrix
    [eigenVec, eigenVals] = eig(covMat);   % Eigen factorization
    eigenVals = diag(eigenVals);
    [~, index] = sort(eigenVals);
    noiseSubspace = eigenVec(:, index(1: sensorNum - sourceNum));
    
    theta = (-30: 0.1: 30)';
    spectrum = zeros(length(theta), 1);
    c = 3e8;
    margin = (c/f)/2;                     % Let distance between 2 adjacent sensors equal to wavelength/2    
    for itr = 1: length(theta)
        steeringVct = exp(-1j*2*pi*f*(margin*(0: sensorNum - 1)'*sind(theta(itr)))/c);
        spectrum(itr) = (steeringVct'*steeringVct)/(steeringVct'*(noiseSubspace*noiseSubspace')*steeringVct);
    end
end