    

    
    if ~exist('TRANS','var')
        TRANS = [1];
    end
    if ~exist('EMIS','var')
        EMIS = [0.3, 0.7];
    end
    if ~exist('n','var')
        n = 1000;
    end
    if ~exist('len','var')
        len = 1000;
    end
    if ~exist('t','var')
        t = 700;
    end
    %collect samples
    [x,~] = hmmgenerate(n,TRANS,EMIS);
    
    xs = unique(x);
    m = length(xs);
    X = zeros(n,len,m);
    tempX = zeros(n,m);
    for i = 1:m
        tempX(x == xs(i), i) = 1;
    end
    X(:,t,:) = tempX;
    
    model = kalmanTD(X)