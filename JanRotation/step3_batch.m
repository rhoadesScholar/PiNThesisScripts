function RMS = runBatch()
    %Batch Settings
    N = 100;

    %Sim Priors (position, object distance, velocity)
    P = [0 0];
    D = [10 10];
    V = [1 0];

    %Sim Settings
    a = 1;
    endT = 100;
    dt = .1;
    sigmaVest = .1;
    sigmaEye = 1;

    for n = 1:N%SHOULD BE PARFOR (if could start pool correctly...)
    err(:,:,:,n) = runSim();
    end
    RMS = sqrt(nanmean(err,4));

    endI = size(Z,3)-1;

    % fig = figure;
    clf

    subplot(2, 2, 1)
    plot(0:dt:endI*dt, squeeze(RMS(1,1,:)), 'LineWidth', 2, 'Color', 'g');
    hold on
    plot(0:dt:endI*dt, squeeze(RMS(1,2,:)), 'LineWidth', 2, 'Color', 'b');
    legend('X', 'Y');
    xlabel('time');
    ylabel('root mean square error');
    title('Position RMSE')

    subplot(2, 2, 3)
    plot(0:dt:endI*dt, squeeze(RMS(2,1,:)), 'LineWidth', 2, 'Color', 'g');
    hold on
    plot(0:dt:endI*dt, squeeze(RMS(2,2,:)), 'LineWidth', 2, 'Color', 'b');
    legend('X', 'Y');
    xlabel('time');
    ylabel('root mean square error');
    title('Object Distance RMSE')

    subplot(2, 2, 4)
    plot(0:dt:endI*dt, squeeze(RMS(3,1,:)), 'LineWidth', 2, 'Color', 'g');
    hold on
    plot(0:dt:endI*dt, squeeze(RMS(3,2,:)), 'LineWidth', 2, 'Color', 'b');
    legend('X', 'Y');
    xlabel('time');
    ylabel('root mean square error');
    title('Velocity RMSE')


    function err = runSim()
    sigmaExpected = [sigmaEye 0; 0 sigmaVest]/dt;
    endI = ceil(endT/dt)+1;

    A = [1 0 dt; 0 1 -dt; 0 0 1];
    C = [0 1 0; 0 0 1];
    muInit = [P; D; V];

    inNoise = gmdistribution(zeros(2), sigmaExpected);

    Z = NaN([size(muInit), endI]);
    Y = NaN([size(C*muInit), endI]);
    K = NaN([size(muInit), endI]);
    Var = NaN([size(muInit,1), size(muInit,1), endI]);
    Mu = NaN([size(muInit), endI]);

    Mu(:,:,1) = muInit;
    Z(:,:,1) = muInit + [sqrt(a*sigmaVest)*[randn(), randn()]; sqrt(a*sigmaEye)*[randn(), randn()]; sqrt(a*sigmaVest)*[randn(), randn()]];
    Var(:,:,1) = [a*sigmaVest, 0, 0; 0, 1/eps, 0; 0, 0, a*sigmaVest];
    for i = 2:endI
        %Get actual latent
        Z(:,:,i)=A*squeeze(Z(:,:,i-1));
        %Get observations
        Y(:,:,i)=C*squeeze(Z(:,:,i))+ones(2).*random(inNoise)';%USES SAME NOISE FOR X & Y

        %Get filter
        thisVar = squeeze(Var(:,:,i-1));
        thisMu = squeeze(Mu(:,:,i-1));
        K(:,:,i) = A*thisVar*A'*C'/(C*A*thisVar*A'*C' + sigmaExpected);
        thisK = squeeze(K(:,:,i));
        Var(:,:,i) = (eye(size(A)) - thisK*C)*A*thisVar*A';
        Mu(:,:,i) = A*thisMu + thisK*(squeeze(Y(:,:,i))-C*A*thisMu);   
    end

    err = (Mu - Z).^2;
    return
    end
end