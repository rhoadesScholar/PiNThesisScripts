function [Y, X, NLP] = autoSequenceMaker(varargin)
    %set options
    for i = 1:2:nargin
        eval([varargin{i} ' = ' mat2str(varargin{i + 1})])
    end
    if ~exist('TRANS','var')
        TRANS = [1];
    end
    if ~exist('EMIS','var')
        EMIS = [0.3, 0.7];
    end
    if ~exist('n','var')
        n = 10;
    end
    
    K = length(EMIS);
    ks = categorical(1:K);

    %collect samples
    [Y,~] = hmmgenerate(n,TRANS,EMIS);
    X = histcounts(categorical(Y), ks);
    
    [NLP] = calculateInfoContent(Y, EMIS, ks);
end