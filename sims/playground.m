function results = playground(machine, learner)
    if ~strcmpi(class(machine), 'stateMachine')
        stateStructure = machine;
        machine = stateMachine;
        machine.stateStruct = stateStructure;
        machine.currentState = 1;
        machine.maxActNum = 2;
    end
    if ~exist('learner', 'var')
        learner = humanPlayer;
    end
    
    [p, fig] = getPlot([]);
    rewards = [];
    acts = learner.nextAct(machine.currentState, 0);
    states = {};
    while ~any(acts == 666)
        [machine, rewards(end+1), states{end+1}] = machine.nextState(acts(end));
        p = getPlot(rewards, p, fig);
        disp(states{end})
        try
            acts(end+1) = learner.nextAct(machine.currentState, rewards(end));
        catch
            break
        end
    end
    results.rewards = rewards;
    results.acts = acts;
    results.states = states;
end

function [p, fig] = getPlot(data, p, fig)
    if isempty(data)
        fig = figure;
        p = plot(0,0,'LineWidth',2);
    else
        p.YData(end+1) = nansum(data);
        p.XData(end+1) = numel(p.XData);
        drawnow
    end
end