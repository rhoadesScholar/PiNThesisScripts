function [Y, X, NLP] = humanSequenceMaker(n)
    %set options
    if ~exist('n', 'var')
        n = 10;
    end

    colors = getColors;

    fig = figure;
    colormap jet
    Y = [];
    Ys = [];
    C = [];
    NLP = NaN(n, 1);
    
    subplot(1, 2, 2);
    nlpPlot = semilogx(NLP,'LineWidth', 2);
    hold on
    xlabel('Draw #')
    ylabel('-log\theta_i')

    subplot(1, 2, 1);
    stim = imagesc(colors(1),[0,max(colors)]);
    
    %collect samples
    while length(Y) < n
        w = waitforbuttonpress;
        if w
            Y(end + 1) = double(get(fig,'CurrentCharacter'));
            c = find(Ys == Y(end));
            if isempty(c)
                Ys(end + 1) = Y(end);
                C(end+ 1) = colors(length(C)+1);
                c = length(C);
            end
            x = histcounts(Y, unique(Y));
            x = x/sum(x);
            alpha = ones(size(x))/length(x) + x;
            alpha(alpha == 0) = eps;
            
            i = length(Y);
            NLP1 = gammaln(i+1) - sum(gammaln(x+1));
            NLP2 = gammaln(sum(alpha)) - sum(gammaln(alpha));
            NLP3 = sum(gammaln(x+alpha)) - gammaln(i+sum(alpha));
            NLP(i) = -(NLP1+NLP2+NLP3);%<-- negative log probability
            
            
            nlpPlot.XData(end+1) = i;
            nlpPlot.YData(end+1) = NLP(i);
            
            stim.CData = (C(c));
        end
    end
    X = histcounts(Y, unique(Y));
    subplot(1,2,1);
    histogram(categorical(Y), categorical(unique(Y)))

%     [NLP] = calculateInfoContent(Y);
end

function colors = getColors
    cs = 0:255;
    colors(1) = cs(1);
    i = 2;
    for t = 0:log2(length(cs))    
        for q = 1:2:2^t
            if ~any(colors == cs(q*length(cs)/2^t))
                colors(i) = cs(q*length(cs)/2^t);
                i = i + 1;
            end
        end
    end
end