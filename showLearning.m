function showLearning(Obs, Vars)
    if ~exist('Vars', 'var') || isempty(Vars)
        for i = 1:size(Obs, 2)
            tempVars = Obs(:, i);
            tempVars = unique(tempVars);
            Vars(i, 1:length(tempVars)) = tempVars;
        end
    end
    
    Dim = size(Vars, 1);
    
    teacher = Leviathan(Vars, Obs);
    learner = Learner(Dim);
    
    Is = NaN(size(Obs, 1), 1);
    Hs = NaN(size(Obs, 1), 1);
    for i = 1:size(Obs, 1)
        x = Obs(i, 1); %assumes X (dependent/output variable) is always first
        [P_ax, Ys_a, m] = teacher.getPax_Ysa(x);
        [P_LAx, P_lx] = getProbLAx(x, Ys_a, m, learner);
        Is(i) = -nansum(P_LAx * P_ax * log2(P_LAx / P_lx));
        Hs(i) = -nansum(P_LAx * P_ax * log2(P_LAx * P_ax));
    end
    Ds = Is ./ Hs;
    
    subplot(2, 2, 1)
    plot(Is, 'LineWidth', 3);
    ylabel('I(L;A)');
    
    subplot(2, 2, 2)
    plot(Hs, 'LineWidth', 3);
    ylabel('H(L,A)');
    
    subplot(2, 2, 3:4)
    plot(Ds, 'LineWidth', 3);
    ylabel('D(L,A)');
end

function [P_LAx, P_x] = getProbLAx(x, Ys_a, m, learner)
    for i = 1:m
        if ~exist('P_x', 'var')||isempty(P_x)
            [P_l, P_x] = learner.getPx_ys([x Ys_a(i, :)]);
            P_LAx = P_LAx + P_l;
        else
            P_LAx = P_LAx + learner.getPx_ys([x Ys_a(i, :)], P_x);
        end
    end
    P_LAx = P_LAx/m;
    return
end