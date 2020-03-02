classdef Agent < handle
    properties
        KMs
        epsilon
        SWs
        CostFun
        Sigmas
        Sigma
        A
        C
        E
        Emahal
        LLike
        opts
    end
    
    methods
        function obj = Agent(varargin)
            props = properties(obj);
            for v = varargin
                obj.(props{1}) = v{1};
                props = props(2:end);
            end
            
            Vars = cat(4, obj.KMs.Vars);
            sigs = @(sigmas) reshape(cell2mat(arrayfun(@(i) nearestSPD(sigmas(:,:,i)), 1:size(sigmas,3), 'UniformOutput', false)), size(sigmas,1), size(sigmas,2), []);
            obj.Sigmas = @(t) sigs(squeeze(Vars(:,:,t,:)));
            obj.Sigma = @(t, k) sigs(squeeze(Vars(:,:,t,k)));
            
            As = {obj.KMs.A};
            obj.A = @(p) nansum(reshape(cell2mat(arrayfun(@(k) As{k}*p(k), 1:length(obj.KMs), 'UniformOutput', false)), size(As{1},1), size(As{1},2), []), 3);          
            Cs = {obj.KMs.C};
            obj.C = @(p) nansum(reshape(cell2mat(arrayfun(@(k) Cs{k}*p(k), 1:length(obj.KMs), 'UniformOutput', false)), size(Cs{1},1), size(Cs{1},2), []), 3);
            
            obj.LLike = @(Y, Mu, Cov) -(logdet(Cov) + log(2*pi)*numel(Y) + ((Y-Mu)'*(Cov\(Y-Mu))))/2;%using log1p and expm1 are hackss, boooo
            obj.E = @(X, p) obj.epsilon*(obj.A(p)*X')';
            obj.Emahal = @(X, p) obj.epsilon * (nansum(reshape(cell2mat(cellfun(@(a, pr) pr*a, cellfun(@(c, a) c*a*c', Cs, As, 'UniformOutput', false), ...
                                                num2cell(p, 2)', 'UniformOutput', false)), size(obj.KMs(1).C,1), [], length(obj.KMs)),3)...
                                                * X')';
            
            if ~any(contains(fields(obj), 'CostFun')) || isempty(obj.CostFun)
                obj.CostFun = 'MSE';
            end
            obj.opts = optimoptions('patternsearch', 'Display','off', 'MaxTime', .0001);%, 'UseVectorized', true
        end
        
        function [SEs, metaMus] = getMetaMus(obj, Mus, Zs, Ys)
            ps = softmax(getLogOdds(Mus));           
            
            switch obj.CostFun
                case 'MSE'
                    x = @(t) ps(:,t)'*squeeze(Mus(:,1:end-1,t));
                    
                    metaMus = arrayfun(@(t) x(t), 1:size(Mus,3), 'UniformOutput', false);
                    metaMus = reshape([metaMus{:}], [], size(metaMus,2));
                    
                case 'Abrupt'                     
                    dist = @(t) gmdistribution(Mus(:,1:end-1,t), obj.Sigmas(t), ps(:,t));
            
                    cdfFunc = @(X, t) diff(cdf(dist(t), [X-obj.E(X,ps(:,t)); X+obj.E(X,ps(:,t))]));            
                    lb = squeeze(max(Mus(:,1:end-1,:),[],1));
                    ub = squeeze(min(Mus(:,1:end-1,:),[],1));            
                    x = @(t) patternsearch(@(X) cdfFunc(X,t), ps(:,t)'*dist(t).mu, [], [], [], [], lb(:,t), ub(:,t), [], obj.opts);
                    
                    metaMus = arrayfun(@(t) x(t), 1:size(Mus,3), 'UniformOutput', false);
                    metaMus = reshape([metaMus{:}], [], size(metaMus,2));
                    
                case 'Mahal'                                       
                    Ysigs = @(t) reshape(cell2mat(arrayfun(@(k) obj.KMs(k).C * obj.Sigma(t, k) * obj.KMs(k).C', 1:length(obj.KMs), 'UniformOutput', false)), size(obj.KMs(1).C,1), [], length(obj.KMs));
                    %^OBJ.SIGMA NEEDS TO BE FIXED TO BE ACTUAL VARIANCE
                    
                    eYs = @(lastMu) reshape(cell2mat(arrayfun(@(k) obj.KMs(k).C * obj.KMs(k).A * lastMu, 1:length(obj.KMs), 'UniformOutput', false)), [], length(obj.KMs))';
                    
                    metaMus = obj.KMs(1).blankMus(1:end-1,:);
                    tempDist = gmdistribution([obj.KMs.muPrior]', obj.Sigmas(1), ps(:,1));
                    cdfFunc = @(X, t) diff(cdf(tempDist, [X-obj.E(X,ps(:,1)); X+obj.E(X,ps(:,1))]));   
                    metaMus(:,1) = patternsearch(@(X) cdfFunc(X,1), ps(:,1)'*tempDist.mu, [], [], [], [], [], [], [], obj.opts);
                    
                    Ydists{1} = gmdistribution(reshape(cell2mat(arrayfun(@(k) obj.KMs(k).C * obj.KMs(k).A * obj.KMs(k).muPrior, 1:length(obj.KMs), 'UniformOutput', false)), [], length(obj.KMs))',...
                                    Ysigs(1), ps(:,1));
                    mahalT = @(t) Ydists{t}.mahal(Ys(:,t)');
                    pMahal = @(t) (exp(-mahalT(t)) ./ nansum(exp(-mahalT(t))))';
                    
                    nextMus = @(lastMus) reshape(cell2mat(arrayfun(@(k) obj.KMs(k).A * lastMus, 1:length(obj.KMs), 'UniformOutput', false)), [], length(obj.KMs))';
                    
                    for i = 2:size(Ys,2)
                        Ydists{i} = gmdistribution(eYs(metaMus(:,i-1)), Ysigs(i), pMahal(i-1));
                        mahalT = @(t) Ydists{t}.mahal(Ys(:,t)');
                        pMahal = @(t) (exp(-mahalT(t)) ./ nansum(exp(-mahalT(t))))';
                        
                        tempDist = gmdistribution(nextMus(metaMus(:,i-1)), obj.Sigmas(i), pMahal(i));%OBJ.SIGMAS NEEDS TO BE FIXED TO BE THE ACTUAL VARIANCE
                        cdfFunc = @(X, t) diff(cdf(tempDist, [X-obj.E(X,pMahal(t)); X+obj.E(X,pMahal(t))]));   
                        metaMus(:,i) = patternsearch(@(X) cdfFunc(X,i), pMahal(i)'*tempDist.mu, [], [], [], [], [], [], [], obj.opts);                        
                    end
            end                      
%                     metaVars = arrayfun(@(t) Sig(t), 1:size(Mus,3), 'UniformOutput', false);
%                     metaVars = reshape([metaVars{:}], size(Sig(1),1), size(Sig(1),2), []);
%             metaVars = cumCov(metaMus');            
%             LEvid = cumsum(arrayfun(@(t) obj.LLike(Ys(:,t), obj.C(ps(:,t))*metaMus(:,t), obj.C(ps(:,t))*metaVars(:,:,t)*obj.C(ps(:,t))'), 1:size(Mus,3)));
            SEs = cat(1,(metaMus - Zs).^2, NaN(1,size(metaMus,2)));%LEvid);
            
            return
        end
       
        function s = logcumsumexp(~, x, w, dim)
            % Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
            switch nargin 
                case 2
                    % Determine which dimension sum will use
                    dim = find(size(x)~=1,1);
                    if isempty(dim), dim = 1; end
                    w = ones(1, size(x,dim));
                    
                case 3
                    if isscalar(w)
                        dim = w;
                        w = ones(1, size(x,dim));
                    else
                        dim = find(size(x)~=1,1);
                        if isempty(dim), dim = 1; end
                    end
                    
                case 4
                    if isempty(w)
                        w = ones(1, size(x,dim));
                    end
            end

            % subtract the largest in each dim
            y = max(x,[],dim);
            x = bsxfun(@minus,x,y);
            s = y + log(cumsum(w.*exp(x),dim));
            i = find(~isfinite(y));
            if ~isempty(i)
                s(i) = y(i);
            end
            return
        end
        
    end
end