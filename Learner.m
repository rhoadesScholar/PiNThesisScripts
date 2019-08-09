classdef Learner < handle
    %Copyright 2019: Jeffrey Rhoades, Harvard University
    %Defined for learning P(x|y1,...yn) where x is the first variable
    %defined in Vars
    
    properties
        Obs%Used for conditional probability table
        Vars
        Dim
    end
    
    methods
        function obj = Learner(dim)
            if ~exist('dim', 'var') || isempty(dim)
                dim = 3;
            end
            obj.Dim = dim;
            obj.Vars = cell(dim, 1);
            for i = 1:dim
                obj.Vars(i) = {'unknown'};
            end
            obj.Obs = 1;%let initiation of learner equate to an observation that there are unknown X | unknown {Y1...Yn}
        end
        
        function [P_l, P_x] = getPx_ys(obj, vars, P_x)
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
            
            P_l = (w_c * P_cond) + (1 - w_c)*P_x;
            
            return
        end
        
        function obj = observe(obj, vars)
            [inds, emptInds, newVars] = obj.getInds(vars);
            
%             inds = cellfun(@(yl, ya) find(strcmp(yl, ya)), obj.Vars, vars);
%             emptInds = isempty(inds);
%             
% %             inds = cell(obj.Dim, 1);
% %             emptInds = cell(obj.Dim, 1);
% %             for i = 1:obj.Dim
% %                 inds(i) = find(cellfun(@(a) strcmp(vars(i), a), obj.Vars(i)));
% %                 emptInds(i) = isempty(inds(i));
% %             end
            
            if any(emptInds)
%                 newVars = vars;
%                 newVars(emptInds) = 'unknown';
%                 obj = obj.observe(newVars);
                obj.observe(newVars);
                for i = emptInds
                    obj.Vars(i, end + 1) = vars(i);
                    inds(i) = length(obj.Vars(i));
                end
            end
            
            cellCall = obj.getCellCall(inds);
            
            try
                eval(sprintf('%s = %s + 1;', cellCall, cellCall));
            catch
                eval(sprintf('%s = 1;', cellCall));
            end
                
        end
        
        function w_c = getCountWeight(obj, inds)
%             [inds, emptInds, ~, newInds] = getInds(obj, ys);
%             
% %             Yl = obj.Vars(2:end);
% %             inds = cellfun(@(yl, ya) find(strcmp(yl, ya)), Yl, ys);
% %             emptInds = isempty(inds);
%             
%             if any(emptInds)
%                 inds = newInds;
% %                 tempVars = ys;
% %                 tempVars(emptInds) = 'unknown';
% %                 inds = cellfun(@(yl, ya) find(strcmp(yl, ya)), Yl, tempVars);
%             end
            
            cellCall = obj.getCellCall([':' num2cell(inds(2:end))]);%IF THIS FAILS: Try inds'
            eval(sprintf('E = nansum(%s, ''all'');', cellCall));
            w_c = 1 - 2^-E;
            return
        end
        
        function [inds, emptInds, newVars, newInds] = getInds(obj, vars)
            if length(vars) < size(obj.Vars,1)
                theseVars = obj.Vars(2:end,:);
            else
                theseVars = obj.Vars;
            end
            inds = arrayfun(@(i) find(strcmp(theseVars(i,:), vars(i))), 1:length(vars));
            emptInds = isempty(inds);
            newVars = vars;
            newVars(emptInds) = {'unknown'};
            newInds = arrayfun(@(i) find(strcmp(theseVars(i,:), vars(i))), 1:length(vars));
            return
        end
        
        function cellCall = getCellCall(obj, inds)
            cellCall = 'obj.Obs(';
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