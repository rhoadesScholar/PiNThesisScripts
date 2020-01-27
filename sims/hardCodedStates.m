
            % stateStruct.probFunction
            % stateStruct.states(i).name
            % stateStruct.states(i).probFunction
            % stateStruct.states(i).rewardFunction
%             % stateStruct.states(i).index --->(array: [i, j, k, ....])
            % stateStruct.states(i).states(j).name ---> (etc.)

machine = stateMachine;
machine.currentState = 1;
probs.rewards.toneA = [.7 .7];
probs.tones = [.5 .5];
probs.rewards.toneB = [.7 .3];
probs.rewards.toneC = [.3 .7];

numPedestals = 2;
numPorts = 2;

%% base state
state.name = 'base'; %sampling from one pedestal or the other
state.probFunction = @(a) [1:numPedestals] == a;
state.rewardFunction = @(a) 0;
state.states = struct();

machine.stateStruct.states = state;

%% 1-tone pedestal
state.name = '1op_pedestal';
state.probFunction = @(a) 1;
state.rewardFunction = @(a) 0;
state.states = struct();

machine.stateStruct.states(1).states = state;

%% tone state for 1-tone pedestal
state.name = 'A_tone';
state.probFunction = @(a) [1:numPorts] == a;
state.rewardFunction = @(a) 0;
state.states = struct();

machine.stateStruct.states(1).states(1).states = state;

%% L port try after toneA
state.name = 'jump[1]';
state.probFunction = @(a) 1;
state.rewardFunction = @(a) rand < probs.rewards.toneA(1);
state.states = struct();

machine.stateStruct.states(1).states(1).states(1).states = state;

%% R port try after toneA
state.name = 'jump[1]';
state.probFunction = @(a) 1;
state.rewardFunction = @(a) rand < probs.rewards.toneA(2);
state.states = struct();

machine.stateStruct.states(1).states(1).states(1).states(2) = state;

%% 2-tone pedestal
state.name = '2op_pedestal';
state.probFunction = @(a) probs.tones;
state.rewardFunction = @(a) 0;
state.states = struct();

machine.stateStruct.states(1).states(2) = state;

%% tone B state
state.name = 'B_tone';
state.probFunction = @(a) [1:numPorts] == a;
state.rewardFunction = @(a) 0;
state.states = struct();

machine.stateStruct.states(1).states(2).states = state;

%% tone C state
state.name = 'C_tone';
state.probFunction = @(a) [1:numPorts] == a;
state.rewardFunction = @(a) 0;
state.states = struct();

machine.stateStruct.states(1).states(2).states(2) = state;

%% L port try after toneB
state.name = 'jump[1]';
state.probFunction = @(a) 1;
state.rewardFunction = @(a) rand < probs.rewards.toneB(a);
state.states = struct();

machine.stateStruct.states(1).states(2).states(1).states = state;

%% R port try after toneB
state.name = 'jump[1]';
state.probFunction = @(a) 1;
state.rewardFunction = @(a) rand < probs.rewards.toneB(a);
state.states = struct();

machine.stateStruct.states(1).states(2).states(1).states(2) = state;

%% L port try after toneC
state.name = 'jump[1]';
state.probFunction = @(a) 1;
state.rewardFunction = @(a) rand < probs.rewards.toneC(a);
state.states = struct();

machine.stateStruct.states(1).states(2).states(2).states = state;

%% R port try after toneC
state.name = 'jump[1]';
state.probFunction = @(a) 1;
state.rewardFunction = @(a) rand < probs.rewards.toneC(a);
state.states = struct();

machine.stateStruct.states(1).states(2).states(2).states(2) = state;