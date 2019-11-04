classdef DirichletLearner < handle
    %Copyright 2019: Jeffrey Rhoades, Harvard University
    %Defined for learning P(x1,...xn)
    
    properties
        Cs%Used for conditional probability table (= sum of occurences)
        Xs
    end
    
    methods
        function obj = DirichletLearner()
            obj.Xs = cell();
            obj.Cs = [];
        end
        
        function [Px] = getPx(obj, vars, P_x)
            [~, ~, ~, newInds] = obj.getInds(vars);
            w_c = obj.getCountWeight(newInds);
            
            condCall = obj.getCellCall(newInds);%IF THIS FAILS: Try newInds'
            setCall = obj.getCellCall([':' num2cell(newInds(2:end))]);
            eval(sprintf('P_cond = nansum(%s, ''all'')/nansum(%s, ''all'');', condCall, setCall));
            
            if ~exist('P_x', 'var')||isempty(P_x)
                tempInds = num2cell(newInds);
                tempInds(2:end) = {':'};
                xCall = obj.getCellCall(tempInds);%IF THIS FAILS: Try newInds'
                tempInds(1) = {':'};
                allCall = obj.getCellCall(tempInds);%IF THIS FAILS: Try newInds'
                eval(sprintf('P_x = nansum(%s, ''all'')/nansum(%s, ''all'');', xCall, allCall));
            end
            
            P_l = nansum([(w_c * P_cond), (1 - w_c)*P_x]);
            
            return
        end
        
        function obj = observe(obj, vars)
            [inds, emptInds, newXs] = obj.getInds(vars);
            
            if any(emptInds)
                obj.observe(newXs);
                for i = find(emptInds)
                    len = sum(cellfun(@(x) ~isempty(x), obj.Xs(i,:)));
                    obj.Xs(i, len + 1) = vars(i);
                    inds{i} = len + 1;
                end
            end
            
            cellCall = obj.getCellCall(inds);
            
            try
                eval(sprintf('%s = %s + 1;', cellCall, cellCall));
            catch
                eval(sprintf('%s = 1;', cellCall));
            end
                
        end        
        
        function [inds, emptInds, newXs, newInds] = getInds(obj, vars)
            if length(vars) < size(obj.Xs,1)
                theseXs = obj.Xs(2:end,:);
            else
                theseXs = obj.Xs;
            end
            inds = arrayfun(@(i) find(strcmp(theseXs(i,:), vars(i))), 1:length(vars), 'UniformOutput', false);
            emptInds = cellfun(@(x) isempty(x), inds);
            newXs = vars;
            newXs(emptInds) = {'unknown'};
            newInds = arrayfun(@(i) find(strcmp(theseXs(i,:), newXs(i))), 1:length(newXs));
            return
        end
        
        function cellCall = getCellCall(obj, inds)
            cellCall = 'obj.Cs(';
            for i = 1:obj.Dim
                if iscell(inds)
                    thisInd = inds{i};
                else
                    thisInd = inds(i);
                end
                if i~= obj.Dim
                    cellCall = [cellCall num2str(thisInd) ', '];
                else
                    cellCall = [cellCall num2str(thisInd) ')'];
                end
            end
            return
        end

    end
end