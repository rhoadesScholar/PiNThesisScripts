function [results, model] = kalmanSensoryTD_ePE(X,r,stage,trial,param)
    
    % initialization
    [N,D] = size(X);
    w = zeros(D,1);         % weights
    eW = zeros(D,1);         % weights for estimated PE
    X = [X; zeros(1,D)];    % add buffer at end
    if nargin < 2 || isempty(r); r = zeros(N,1); end
    
    % parameters
    if nargin < 5 || isempty(param); param = KTD_defparam; end
    C = param.c*eye(D); % prior covariance
    s = param.s;        % noise variance
    g = param.g;        % discount factor
    
    if length(s)==1; s = zeros(N,1)+s; end
    if length(param.q) ==1; param.q = zeros(N,1)+param.q; end
    if param.TD && length(param.lr)==1; param.lr = zeros(N,1)+param.lr; end
    
    % run Kalman filter
    lastTrial = 0;
    figure
    colormap jet
    for n = 1:N
        
        % store results
        model(n).w = w;
        model(n).eW = eW;
        model(n).V = X(n,:)*w;      % value estimate
        
        Q = param.q(n)*eye(D);      % transition covariance
        h = X(n,:) - g*X(n+1,:);    % temporal difference features
        rhat = h*w;
        eDt = h*eW;
        dt = r(n) - rhat;           % prediction error
        mDt = dt - eDt;           % meta prediciton error
        C = C + Q;                  % a priori covariance
        P = h*C*h'+s(n);            % residual covariance
        K = C*h'/P;                 % Kalman gain
        if param.TD
            w = w + param.lr(n)*h'*dt;
            eW = eW + param.lr(n)*h'*eDt;
        else
            w = w + K*dt;           % weight update
            eW = eW + K*mDt;           % weight update
        end
        C = C - K*h*C;              % posterior covariance update
        
        model(n).C = C;
        model(n).K = K;
        model(n).dt = dt;
        model(n).mDt = mDt;
        model(n).rhat = rhat;
        model(n).eDt = eDt;
        
        if trial(n) ~= lastTrial
            lastTrial = trial(n);
            i = 1;
            
            if exist('im', 'var')
                im.CData = C;
                drawnow expose
            else
                im = imagesc(C);
                colorbar
            end
        else
            i = i + 1;
        end
        
        try
            results(stage(n)).rhat(trial(n), i) = rhat;
            results(stage(n)).dt(trial(n), i) = dt;            
            results(stage(n)).mDt(trial(n), i) = mDt;
            results(stage(n)).eDt(trial(n), i) = eDt;
        catch
            results(stage(n)).rhat = rhat;
            results(stage(n)).dt = dt;          
            results(stage(n)).mDt = mDt;
            results(stage(n)).eDt = eDt;
        end
    end
    close