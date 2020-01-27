function errs = runBatchSim(argStr)
    %Batch Settings
    N = 1000;

    %Sim Priors (position, object distance, velocity)
    P = [0 0];
    D = [10 10];
    M = [0 0];
    V = [1 0];

    %Sim Settings
    a = 1;
    endT = 500;
    dt = .5;
    sigmaVest = 1;
    sigmaEye = 1;
    objInitPosVar = 100;
    objInitMovVar = 50;
    
    %Plot Serrings
    color = 'b';
    
    eval(argStr);
    
    sigmaExpected = [sigmaEye 0; 0 sigmaVest]/dt;
    endI = ceil(endT/dt)+1;

    A = [1 0 0 dt; 0 1 dt -dt; 0 0 1 0; 0 0 0 1];
    C = [0 1 0 0; 0 0 0 1];
    muInit = [P; D; M; V];

    inNoise = gmdistribution([0 0], sigmaExpected);
    
    tic
    errs = NaN([size(muInit,1),ceil(endT/dt)+1,N]);
    for n = 1:N%SHOULD BE PARFOR (if could start pool correctly...)
        oldMu = muInit;
        oldVar = [a*sigmaEye, 0, 0, 0; 0, a*objInitPosVar, 0, 0; 0, 0, a*objInitMovVar, 0; 0, 0, 0, a*sigmaVest];
        errs(:,:,n) = runSim();
    end
    RMSE = sqrt(nanmean(errs,3));
    sems = std(errs,0,3)/sqrt(N);
    toc
    
    plotRMSE();
    
    toc
    
    function err = runSim()
        err = NaN([size(muInit,1),ceil(endT/dt)+1]);
        for v = 1:numel(muInit)
            oldZ = muInit + [sqrt(a*sigmaEye)*[randn(), randn()]; sqrt(a*objInitPosVar)*[randn(), randn()]; sqrt(a*objInitMovVar)*[randn(), randn()]; sqrt(a*sigmaVest)*[randn(), randn()]];
        err(:,1) = sum((muInit - oldZ).^2,2); %FIRST ENTRY IS RMSE BEFORE FIRST UPDATE
        for i = 2:endI
            %Get actual latent
            oldZ = A*oldZ;
            %Get observations
            Y = C*oldZ + random(inNoise,2)';

            %Get filter
            K = A*oldVar*A'*C'/(C*A*oldVar*A'*C' + sigmaExpected);
            oldVar = (eye(size(A)) - K*C)*A*oldVar*A';
            oldMu = A*oldMu + K*(Y-C*A*oldMu);   
            err(:,i) = sum((oldMu - oldZ).^2,2);
        end

        return
    end

    function plotRMSE()
        endI = size(RMSE,2)-1;

        subplot(2, 2, 1:2)
        plotshade(RMSE(1,:), sems(1,:), 0:dt:endI*dt, .3, color, 2);
        hold on
        xlabel('time');
        ylabel('root mean square error');
        title('Position RMSE')

        subplot(2, 2, 3)
        plotshade(RMSE(2,:), sems(2,:), 0:dt:endI*dt, .3, color, 2);
        hold on
        xlabel('time');
        ylabel('root mean square error');
        title('Object Distance RMSE')

        subplot(2, 2, 4)
        plotshade(RMSE(3,:), sems(3,:), 0:dt:endI*dt, .3, color, 2);
        hold on
        xlabel('time');
        ylabel('root mean square error');
        title('Velocity RMSE')
        return
    end
end