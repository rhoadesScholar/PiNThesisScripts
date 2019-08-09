function showLearning(Obs, setNames)
    for o = 1:size(Obs, 3)
        obs = Obs(:, :, o);
        for i = 1:size(obs, 2)
            tempVars = obs(:, i);
            tempVars = unique(tempVars);
            Vars(i, 1:length(tempVars)) = tempVars;
        end

        Dim = size(Vars, 1);

        teacher = Leviathan(Vars, obs);
        learner = Learner(Dim);

        Is = NaN(size(obs, 1), 1);
        Hs = NaN(size(obs, 1), 1);
        for i = 1:size(obs, 1)
            x = obs(i, 1); %assumes X (dependent/output variable) is always first
            [P_ax, Ys_a, m] = teacher.getPax_Ysa(x);
            [P_LAx, P_lx] = getProbLAx(Ys_a, m, learner);
            Is(i) = -nansum(P_LAx * P_ax * log2(P_LAx / P_lx));
            Hs(i) = -nansum(P_LAx * P_ax * log2(P_LAx * P_ax));
            learner.observe(obs(i, :));
        end
        Ds = Is ./ Hs;
        
        subplot(2, 2, 1)
        plot(Is, 'LineWidth', 3);
        ylabel('I(L;A)');
        hold on;

        subplot(2, 2, 2)
        plot(Hs, 'LineWidth', 3);
        ylabel('H(L,A)');
        hold on;

        subplot(2, 2, 3:4)
        plot(Ds, 'LineWidth', 3);
        ylabel('D(L,A)');
        hold on;
    end
    
    if exist('setNames', 'var') && ~isempty(setNames)
        legend(setNames)
    end
end

function [P_LAx, P_x] = getProbLAx(vars, m, learner)
    for i = 1:m
        if ~exist('P_x', 'var')||isempty(P_x)
            [P_l, P_x] = learner.getPx_ys(vars(i, :));
            P_LAx = P_l;
        else
            P_LAx = P_LAx + learner.getPx_ys(vars(i, :), P_x);
        end
    end
    P_LAx = P_LAx/m;
    return
end