function [NLP] = calculateInfoContent(Y, EMIS, ks)
    %calculate information content of seq
    n = length(Y);
    NLP = NaN(n, 1);
    oNLP = NaN(n, 1);
    for i = 1:n
        try 
            x = histcounts(categorical(Y(1:i)), ks);
            j = find(ks == categorical(Y(i)));
        catch
            x = histcounts(Y(1:i), unique(Y(1:i)));
            j = find(unique(Y(1:i)) == Y(i));
        end
%         x = x/sum(x);
        try
            alpha = EMIS + x;
        catch
            alpha = ones(size(x))/length(x) + x;
        end
        alpha(alpha == 0) = eps;
        
        NLP1 = gammaln(i+1) - sum(gammaln(x+1));
        NLP2 = gammaln(sum(alpha)) - sum(gammaln(alpha));
        NLP3 = sum(gammaln(x+alpha)) - gammaln(i+sum(alpha));
        NLP(i) = -(NLP1+NLP2+NLP3);%<-- negative log probabilitye
        
        %other?
        NLP1 = gammaln(i+1) - sum(gammaln(x(j)+1));
        NLP2 = gammaln(sum(alpha)) - sum(gammaln(alpha));
        NLP3 = sum(gammaln(x(j)+alpha(j))) - gammaln(i+sum(alpha));
        oNLP(i) = -(NLP1+NLP2+NLP3);
    end
    semilogx(NLP,'LineWidth', 2)
    hold on
    xlabel('Draw #')
    ylabel('-log\theta_i')
    semilogx(oNLP,'LineWidth', 2)
    drawnow
end