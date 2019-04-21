function sourceNum = AIC(snapshots, sensorNum, eigVals)
%---sourceNum: the number of sources using AIC method---%
%---snapshots: the number of snapshots------------------%
%---sensorNum: the number of array elements-------------% 
%---eigVals: eigen values with descendant sort--------%
    t = inf;                
    for sourceNum = 1: sensorNum - 1
        num = 0;
        den = 1;
        for iter = sourceNum + 1: sensorNum
            num = num + eigVals(iter);
            den = den * eigVals(iter);
        end
        Lambda = (num/(sensorNum - sourceNum)) / den^(1/(sensorNum - sourceNum));
        etrpy = 2*snapshots*(sensorNum - sourceNum)*log(Lambda) + ...
                2*sourceNum*(2*sensorNum - sourceNum);    % AIC entropy
%-------find minimum loss entropy----------------%
        if etrpy > t
            break;
        else
            t = etrpy;
        end
    end
    sourceNum = sourceNum - 1;          % fix n
end