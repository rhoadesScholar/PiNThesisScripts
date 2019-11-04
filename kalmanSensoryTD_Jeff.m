function [results, model] = kalmanSensoryTD_Jeff(X,r,stage,trial,param)
    
    % initialization
    [N,D] = size(X);
    w = zeros(D,1);         % weights
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
    xhat = zeros(size(X(1,:)))';
    for n = 1:N        
        % store results
        model(n).w = w;
        model(n).V = X(n,:)*w;      % value estimate
        
        Q = param.q(n)*eye(D);      % transition covariance
        h = diag(X(n,:)' - xhat);%g*X(n+1,:);    % temporal difference features
        rhat = w'*xhat;
        dt = r(n) - rhat;           % prediction error
        C = C + Q;                  % a priori covariance
        P = h*C*h'+s(n);            % residual covariance
        if rcond(P) == 0 
            P = 1;
        end
        #####SOFTMAX
        K = C*h'/P;                 % Kalman gain
        w0 = w;
        if param.TD
            w = w + param.lr(n)*h'*dt;
        else
            w = w + K*dt;           % weight update
        end
        C = C - K*h*C;              % posterior covariance update
        if ~any(any(isnan(K)))
            xhat = xhat + K*(X(n,:)'-h*xhat);
        end
        
        model(n).C = C;
        model(n).K = K;
        model(n).dt = dt;
        model(n).rhat = rhat;
        
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
        catch
            results(stage(n)).rhat = rhat;
            results(stage(n)).dt = dt;
        end
    end
    close