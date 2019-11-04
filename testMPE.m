
ePE = @(i,a) i*sin(0:.1:pi);
aPE = @(i,a) a*sin(0:.1:pi);

% mPE1 = @(i,a) iPE(i,a)-aPE(i,a)
% mPE2 = @(i,a) abs(iPE(i,a))-abs(aPE(i,a))
mPE3 = @(i,a) abs(ePE(i,a)-aPE(i,a))
% mPE4 = @(i,a) abs(iPE(i,a))-aPE(i,a)
% mPE5 = @(i,a) iPE(i,a)-abs(aPE(i,a))

funcs = whos;
funcnames = {funcs(strcmp({funcs.class},'function_handle')).name};
clear funcs
cellfun(@(s,i) evalin('base', sprintf("funcs{%i} = %s;", i, s)), funcnames, num2cell(1:length(funcnames)));



is = [3, 1, -3, 1, 3, -3, -3, 3];%internal prediction of prediction error
as = [1, 3, 1, -3, -3, 3, -3, 3];%actual prediction error

for n = 1:max([length(is), length(is)])
        fig = figure;
        hold on
        
        if length(is) > length(as)
            as = repmat(as, 1, ceil(length(is)/length(as)));
        elseif length(as) > length(is)
            is = repmat(is, 1, ceil(length(as)/length(is)));
        end
        
        i = is(n); a = as(n);
        
        for f = 1:length(funcs)
            plot(funcs{f}(i,a), 'LineWidth', 2)
        end

    legend(funcnames)
end
