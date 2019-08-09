classdef Leviathan < handle
    %Copyright 2019: Jeffrey Rhoades, Harvard University
    %Defined for knowing P(x|y1,...yn) where x is the first variable
    %defined in Vars
    
    properties
        Obs%Used for conditional probability table
        Vars
        Dim
    end
    
    methods
        function obj = Leviathan(Vars, Obs)
            obj.Dim = size(Vars, 1);
            obj.Vars = Vars;
            for i = 1:size(Obs, 1)
                obj.observe(Obs(i, :));
            end
        end
        
        function [P_a, Ys_a, m] = getPax_Ysa(obj, x)
            ind = find(strcmp(obj.Vars(1, :), x));
            
            tempInd = num2cell(ind*ones(1, obj.Dim));
            tempInd(2:end) = {':'};
            xCall = obj.getCellCall(tempInd);%IF THIS FAILS: Try newInds'
            tempInd(1) = {':'};
            allCall = obj.getCellCall(tempInd);%IF THIS FAILS: Try newInds'
            eval(sprintf('P_a = nansum(%s, ''all'')/nansum(%s, ''all'');', xCall, allCall));
            eval(sprintf('xInds = find(%s > 0);', xCall));
            
            subCall = obj.getSubCall();
            eval(sprintf('[%s]  = ind2sub(size(P_a), find(P_a > 0));', subCall));
            
            m = length(sub(1,:));
            Ys_a = cell(m, obj.Dim);
            for i = 1:m
                subCall = obj.getSubCall(i);
                eval(sprintf('theseInds = [%s];', subCall));
                theseYs = cell(1, obj.Dim);
                for j = 1:obj.Dim
                    theseYs(j) = obj.Vars(j, theseInds(j));
                end
                Ys_a(i, :) = theseYs;
            end
            
            return
        end
        
        function obj = observe(obj, vars)
            [inds, emptInds] = obj.getInds(vars);
            
            if any(emptInds)
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
        
        function subCall = getSubCall(obs, ind)
            subCall = '';
            if ~exist('ind', 'var') || isempty(ind)
                ind = ':';
            else
                ind = num2str(ind);
            end
            for i = 1:obs.Dim
                if i == obs.Dim
                    subCall = sprintf('%ssub(%i,%s)', subCall, i, ind);
                else
                    subCall = sprintf('%ssub(%i,%s), ', subCall, i, ind);
                end
            end
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