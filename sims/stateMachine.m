classdef stateMachine
    properties
       currentState % array: [i, j, k, ....]
       stateStruct
       maxActNum
            % stateStruct.probFunction
            % stateStruct.states(i).name
            % stateStruct.states(i).probFunction
            % stateStruct.states(i).rewardFunction
%             % stateStruct.states(i).index --->(array: [i, j, k, ....])
            % stateStruct.states(i).states(j).name ---> (etc.)
    end
    methods
        function [obj, reward, name] = nextState(obj, act)
            if ~exist('act', 'var')
                act = [];
            end
                act = mod(act-1,obj.maxActNum)+1;
                thisState = obj.getState(obj.currentState);
                probs = thisState.probFunction(act);
                nextInd = find(probs == nanmax(probs));
                if numel(nextInd) > 1
                    nextInd = nextInd(randperm(numel(nextInd), 1));
                end
                reward = thisState.states(nextInd).rewardFunction(act);
                obj.currentState = [obj.currentState, nextInd];
                
                if contains(thisState.states(nextInd).name, 'jump')
                    if contains(thisState.states(nextInd).name, 'next')
                        [obj, reward2, ~] = nextState(obj, act);
                        reward = reward + reward2;
                    else
                       obj.currentState = str2num(erase(thisState.states(nextInd).name, 'jump'));
                       reward = reward + obj.getState(obj.currentState).rewardFunction(act); 
                    end
                end
                
                name = obj.getState(obj.currentState).name;
        end
        
        function state = getState(obj, index)
            if isempty(index), index = 1; end
            state = obj.stateStruct;
            for i = 1:numel(index)
                state = state.states(index(i));
            end
        end
    end
end
