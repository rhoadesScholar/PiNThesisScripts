function [results, model, missTrials] = preconKTD_sim(preconProb, showStages, varCon, param)
    
    if nargin < 4; param = KTD_defparam; end

    %sensory preconditioning
    nTrials = 10;
    trial_length = 10;
    dur = 2;
    ons = 3;
    basis = 'CSC';
    if ~exist('preconProb', 'var') || isempty(preconProb)
        preconProb = param.preconProb;
    end
    if ~exist('varCon', 'var') || isempty(preconProb)
        varCon = false;
    end

    % x = [A,B, M,N, X,Y]

    % AB
    % 100% certainty - rewarded
    x1 = zeros(trial_length,6);
    x1(:,1) = KTD_make_stimulus(ons,dur,trial_length);
    x1(:,2) = KTD_make_stimulus(ons+dur,dur,trial_length);
    f1 = construct_basis(basis,x1);
    r1 = zeros(trial_length,1);            
    F1 = repmat(f1,nTrials,1);
    R1 = repmat(r1,nTrials,1);
    trialTemp = cell2mat(arrayfun(@(x) x*ones(trial_length,1), 1:nTrials, 'UniformOutput', false));
    trialTemp = trialTemp(:);
    trial = trialTemp;
    stage = ones(size(F1,1));

    % MN
    % variable certainty - rewarded
    x2 = zeros(trial_length,6);
    x2(:,3) = KTD_make_stimulus(ons,dur,trial_length);
    x2miss = x2;
    x2(:,4) = KTD_make_stimulus(ons+dur,dur,trial_length);
    f2 = construct_basis(basis,x2);
    f2miss = construct_basis(basis,x2miss);
    inds = randperm(nTrials);
    missTrials = inds(1:round((1-preconProb)*nTrials));
    r2 = zeros(trial_length,1);
    F2 = [];
    for i = 1:nTrials
        if any(missTrials == i)
           F2 = [F2; f2miss];
        else
           F2 = [F2; f2];
        end
    end
    R2 = repmat(r2,nTrials,1);
    trial = [trial; trialTemp];
    stage = [stage; 2*ones(size(F2,1))];

    % XY
    % 100% certainty - unrewarded
    x3 = zeros(trial_length,6);
    x3(:,5) = KTD_make_stimulus(ons,dur,trial_length);
    x3(:,6) = KTD_make_stimulus(ons+dur,dur,trial_length);
    f3 = construct_basis(basis,x3);
    r3 = zeros(trial_length,1);            
    F3 = repmat(f3,nTrials,1);
    R3 = repmat(r3,nTrials,1);
    trial = [trial; trialTemp];
    stage = [stage; 3*ones(size(F3,1))];

    % B+
    % conditioning
    x4 = zeros(trial_length,6);
    x4(:,2) = KTD_make_stimulus(ons+dur,dur,trial_length);
    f4 = construct_basis(basis,x4);
    r4 = KTD_make_stimulus(ons+2*dur-1,1,trial_length);            
    F4 = repmat(f4,nTrials,1);
    R4 = repmat(r4,nTrials,1);
    trial = [trial; trialTemp];
    stage = [stage; 4*ones(size(F4,1))];

    % N+
    % conditioning
    x5 = zeros(trial_length,6);
    x5(:,4) = KTD_make_stimulus(ons+dur,dur,trial_length);
    f5 = construct_basis(basis,x5);
    r5 = KTD_make_stimulus(ons+2*dur-1,1,trial_length);            
    F5 = repmat(f5,nTrials,1);
    if varCon
        inds = randperm(nTrials);
        missTrials = inds(1:round((1-preconProb)*nTrials));
        r5miss = zeros(trial_length,1);
        R5 = [];
        for i = 1:nTrials
            if any(missTrials == i)
               R5 = [R5; r5miss];
            else
               R5 = [R5; r5];
            end
        end
    else
        R5 = repmat(r5,nTrials,1);
    end
    trial = [trial; trialTemp];
    stage = [stage; 5*ones(size(F5,1))];

    % Y-
    % conditioning
    x6 = zeros(trial_length,6);
    x6(:,6) = KTD_make_stimulus(ons+dur,dur,trial_length);
    f6 = construct_basis(basis,x6);
    r6 = zeros(trial_length,1);            
    F6 = repmat(f6,nTrials,1);
    R6 = repmat(r6,nTrials,1);
    trial = [trial; trialTemp];
    stage = [stage; 6*ones(size(F6,1))];

    % Probe trial: A
    x7 = zeros(trial_length,6);
    x7(:,1) = KTD_make_stimulus(ons,dur,trial_length);
    f7 = construct_basis(basis,x7);
    r7 = zeros(trial_length,1);            
    F7 = repmat(f7,nTrials,1);
    R7 = repmat(r7,nTrials,1);
    trial = [trial; trialTemp];
    stage = [stage; 7*ones(size(F7,1))];

    % Probe trial: M
    x8 = zeros(trial_length,6);
    x8(:,3) = KTD_make_stimulus(ons,dur,trial_length);
    f8 = construct_basis(basis,x8);
    r8 = zeros(trial_length,1);            
    F8 = repmat(f8,nTrials,1);
    R8 = repmat(r8,nTrials,1);
    trial = [trial; trialTemp];
    stage = [stage; 8*ones(size(F8,1))];

    % Probe trial: X
    x9 = zeros(trial_length,6);
    x9(:,5) = KTD_make_stimulus(ons,dur,trial_length);
    f9 = construct_basis(basis,x9);
    r9 = zeros(trial_length,1);            
    F9 = repmat(f9,nTrials,1);
    R9 = repmat(r9,nTrials,1);
    trial = [trial; trialTemp];
    stage = [stage; 9*ones(size(F9,1))];
    
%     % Probe trial: B
%     x10 = zeros(trial_length,6);
%     x10(:,2) = KTD_make_stimulus(ons,dur,trial_length);
%     f10 = construct_basis(basis,x10);
%     r10 = zeros(trial_length,1);            
%     F10 = repmat(f10,nTrials,1);
%     R10 = repmat(r10,nTrials,1);
%     trial = [trial; trialTemp];
%     stage = [stage; 10*ones(size(F10,1))];
%     
%     % Probe trial: N
%     x11 = zeros(trial_length,6);
%     x11(:,4) = KTD_make_stimulus(ons,dur,trial_length);
%     f11 = construct_basis(basis,x11);
%     r11 = zeros(trial_length,1);            
%     F11 = repmat(f11,nTrials,1);
%     R11 = repmat(r11,nTrials,1);
%     trial = [trial; trialTemp];
%     stage = [stage; 11*ones(size(F11,1))];
%     
%     % Probe trial: Y
%     x12 = zeros(trial_length,6);
%     x12(:,6) = KTD_make_stimulus(ons,dur,trial_length);
%     f12 = construct_basis(basis,x12);
%     r12 = zeros(trial_length,1);            
%     F12 = repmat(f12,nTrials,1);
%     R12 = repmat(r12,nTrials,1);
%     trial = [trial; trialTemp];
%     stage = [stage; 12*ones(size(F12,1))];
    
    F = [F1; F2; F3; F4; F5; F6; F7; F8; F9];% F10; F11; F12];
    r = [R1; R2; R3; R4; R5; R6; R7; R8; R9];% R10; R11; R12];

    [results, model] = kalmanTD_Jeff(F,r,stage,trial,param);
    stageNum = length(results);
%     for n=1:length(model)
%         w(n,:)=model(n).w;
%         c(n,:)=model(n).C(ons,:);
%         k(n,:)=model(n).K;
%     end
%     results.model = model;
%     results.W = w(1:trial_length:end,trial_length+1);
%     results.C = c(3:trial_length:end,:);
%     results.K = k(3:trial_length:end,:);
    
    if ~exist('showStages', 'var')
        showStages = 1:stageNum;
    end
    plotResults(results, showStages)
end
    
function plotResults(results, showStages)
    colors = linspecer(length(showStages));
    rhatPlot = figure;
    dtPlot = figure;
    for s = showStages%stage
%         labels(end+1) = {sprintf('Stage #%i', s)};
        for t = 1:size(results(s).rhat, 1)%trial
            figure(rhatPlot);
            hold on;        
            p = plot(results(s).rhat(t,:), 'Color', colors(showStages==s,:), 'DisplayName', sprintf('Stage #%i', s));
            if t ~= 1
                p.Annotation.LegendInformation.IconDisplayStyle = 'off';
            end            
        end
        
            
        for t = 1:size(results(s).dt, 1)%trial
            figure(dtPlot);
            hold on;        
            p = plot(results(s).dt(t,:), 'Color', colors(showStages==s,:), 'DisplayName', sprintf('Stage #%i', s));
            if t~=1
                p.Annotation.LegendInformation.IconDisplayStyle = 'off';
            end
        end
%         plot(nanmean(results(s).dt(:,:),1), 'Color', colors(s,:), 'LineWidth', 2);        
    end
    figure(rhatPlot);
    title('Reward Prediction');
    legend
    axis tight
    
    figure(dtPlot);
    title('Prediction Error')
    legend
    axis tight
end