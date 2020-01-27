function trainStats = mdpSim(type, varargin)
    if ~exist('type','var')
        type = '1vC';
    end
    for v = 1:2:nargin-1
        eval(sprintf('%s = %s', varargin{v}, varargin{v+1}));
    end
    
    singleToneProb = .7;
    Params.Gain = 0.01;
    Params.Initial = 0;
    Params.Max = 0.9;
    Params.MaxCount = (Params.Max-Params.Initial)/Params.Gain;
    cueAProb = .8;
    cueBProb = .8;

    if strcmpi(type,'1vC')
        %% two-cue vs. static 1-cue
        mdp = createMDP(["start" "singleTonePedestal" "cueA" "cueB" "reward" "punish"],...
            ["left" "right"]);
        mdp.T(1,2,1) = 1;
        mdp.T(1,3:4,2) = .5;

        mdp.T(2,5:6,1) = [singleToneProb, 1-singleToneProb];
        mdp.T(2,5:6,2) = [singleToneProb, 1-singleToneProb];

        mdp.T(3,5:6,1) = [cueAProb, 1-cueAProb];
        mdp.T(3,5:6,2) = [1-cueAProb, cueAProb];

        mdp.T(4,5:6,1) = [1-cueBProb, cueBProb];
        mdp.T(4,5:6,2) = [cueBProb, 1-cueBProb];

        mdp.R(2:4,5,1:2) = 1;

        mdp.TerminalStates = ["reward";"punish"];

        env = rlMDPEnv(mdp);
        env.ResetFcn = @() 1;   
    elseif strcmpi(type,'1vM')
        %% static 1-cue vs. momentum 1-cue
        mdp = createMDP(["start", "reward", "punish"],...
            ["staticChoice" "momentumChoice"]);
        mdp.T(1,2:3,1) = [singleToneProb 1-singleToneProb];
        mdp.T(1,2,2) = 1;
        
        mdp.R(1,2,1:2) = [1 Params.Initial];
        

        mdp.TerminalStates = ["reward";"punish"];

        env = rlMDPEnv_Jeff(mdp);
        env.ResetFcn = @(this) momentumResetFunction(this, Params);
        env.StepFcn = @(this, Action, simCount) momentumStepFunction(this, Action, simCount, Params);
        
    elseif strcmpi(type,'MvC')
        %% two-cue vs. momentum 1-cue
        mdp = createMDP(["start" "momentumChoice" "cueA" "cueB" "reward" "punish"],...
            ["left" "right"]);
        mdp.T(1,2,1) = 1;
        mdp.T(1,3:4,2) = .5;

        mdp.T(2,5,1:2) = 1;

        mdp.T(3,5:6,1) = [cueAProb, 1-cueAProb];
        mdp.T(3,5:6,2) = [1-cueAProb, cueAProb];

        mdp.T(4,5:6,1) = [1-cueBProb, cueBProb];
        mdp.T(4,5:6,2) = [cueBProb, 1-cueBProb];

        mdp.R(3:4,5,1:2) = 1;
        mdp.R(2,5,1:2) = Params.Initial;

        mdp.TerminalStates = ["reward";"punish"];

        env = rlMDPEnv_Jeff(mdp);
        env.ResetFcn = @(this) momentumResetFunction(this, Params);
        env.StepFcn = @(this, Action, simCount) momentumStepFunction(this, Action, simCount, Params);
    end
    %% make agent
    qTable = rlTable(getObservationInfo(env),getActionInfo(env));
    critic = rlRepresentation(qTable);
    critic.Options.LearnRate = 1;
    
    opt = rlQAgentOptions;
    opt.EpsilonGreedyExploration.Epsilon = 0.9;
    opt.EpsilonGreedyExploration.EpsilonMin = 0.001;
    opt.EpsilonGreedyExploration.EpsilonDecay = 0.01;
    opt.DiscountFactor = 1;

    agent = rlQAgent(critic,opt);
    attachLogger(agent, 9999999);
    

    trainOpts = rlTrainingOptions;

    trainOpts.MaxEpisodes = 2000;
    trainOpts.ScoreAveragingWindowLength = 100;
    trainOpts.Verbose = true;
    trainOpts.Plots =   "training-progress";

    trainStats = train(agent,env,trainOpts);
    
    states = cell2mat(arrayfun(@(i) trainStats.Experiences(i).Observation.MDPObservations.Data, 1:numel(trainStats.Experiences), 'UniformOutput', false));
    states = env.Model.idx2state(states);
    trainStats.States = states;
    trainStats.Model = env;
    
    showRLresults
end

function momentumStepFunction(this, Action, simCount, Params)
    rewInd = this.Model.state2idx("reward");
    punInd = this.Model.state2idx("punish");
    if strcmpi(Action, "momentumChoice")
        RT = this.Model.R;
        temp = zeros(size(RT));
        temp(1, rewInd, this.Model.action2idx(Action)) = 1;
        RT = RT + temp*Params.Gain;
        this.Model.setRewardTransition(min(RT,temp));
        % NEED TO MAKE SCENARIO RESET
    elseif strcmpi(this.getCurrentState, "momentumChoice")
        momInd = this.Model.state2idx("momentumChoice");
        RT = this.Model.R;
        RT(momInd, rewInd, :) = RT(momInd, rewInd, :) + Params.Gain;
        temp = (RT > 0)*Params.Max;
        this.Model.setRewardTransition(min(RT,temp));
        
        if mod(simCount, Params.MaxCount) == 0  %reset scenario
            %switch cue pairing
            cues = this.Model.States(contains(this.Model.States, 'cue'));  
            cueInds = this.Model.state2idx(cues);
            cuePairSwitch = randi(numel(cues)+1);
            switch cuePairSwitch
                case numel(cues)+1
                    this.Model.T(cueInds,[rewInd punInd],:) = fliplr(this.Model.T(cueInds,[rewInd punInd],:));
                otherwise
                    this.Model.T(cueInds(cuePairSwitch),[rewInd punInd],:) = fliplr(this.Model.T(cueInds(cuePairSwitch),[rewInd punInd],:));
            end
            
            %switch action leading to options
            if rand > 0.5
                this.Model.T(1,[momInd; cueInds],:) = flip(this.Model.T(1,[momInd; cueInds],:),3);
            end
            
            %reset momentum option
            temp = this.Model.R;    
            temp(momInd, rewInd, :) = Params.Initial;
            this.Model.setRewardTransition(temp);

        end
    end
end

function State = momentumResetFunction(this,Params)
    State = 1;
end