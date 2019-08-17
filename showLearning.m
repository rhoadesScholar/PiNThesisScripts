function data = showLearning(Obs, setNames, before, silent)
    fig = figure;
    if ~exist('before', 'var')||isempty(before)
        before = false;
    end
    if ~exist('silent', 'var')||isempty(silent)
        silent = true;
    end
    for o = 1:size(Obs, 1)
        tic
        obs = squeeze(Obs(o, :, :));
        Vars = cell(size(obs, 2), 1);
        for i = 1:size(obs, 2)
            tempVars = obs(:, i);
            tempVars = unique(tempVars);
            Vars(i, 1:length(tempVars)) = tempVars;
        end

        teacher = Leviathan(Vars, obs);
        learner = Learner(teacher.Dim);

        bIs = NaN(size(obs, 1), 1);
        bHs = NaN(size(obs, 1), 1);
        aIs = NaN(size(obs, 1), 1);
        aHs = NaN(size(obs, 1), 1);
        U = NaN(size(obs, 1), 1);
        f = waitbar(0, sprintf('Learning (%i/%i)...', o, size(Obs, 1)));
%         Inds = randperm(size(obs, 1));
%         for i = Inds
        for i = 1:size(obs, 1)
%             waitbar(find(Inds == i)/length(Inds), f);
            waitbar(i/size(obs, 1), f);
            if i ==1
%             if i == Inds(1)
                [P_ax, P_lx, P_LAx] = getProbs(teacher, learner);
                bIs(i) = nansum(P_LAx .* P_ax .* log2(P_LAx ./ P_lx));
                bHs(i) = -nansum(P_LAx .* P_ax .* log2(P_LAx .* P_ax));
            else
                bIs(i) = aIs(i-1);
                bHs(i) = aHs(i-1);
            end
            
            learner.observe(obs(i, :));
            
            [P_ax, P_lx, P_LAx] = getProbs(teacher, learner);            
            aIs(i) = nansum(P_LAx .* P_ax .* log2(P_LAx ./ P_lx));
            aHs(i) = -nansum(P_LAx .* P_ax .* log2(P_LAx .* P_ax));
            U(i) = nansum(log2(P_LAx));
        end
        bDs = 1 - (bIs ./ bHs);
        aDs = 1 - (aIs ./ aHs);
        
        %% plot
        if before
            Is = bIs;
            Hs = bHs;
            Ds = bDs;
        else
            Is = aIs;
            Hs = aHs;
            Ds = aDs;
        end
        
        figure(fig)
        
        subplot(2, 4, 1)
        plot(Is, 'LineWidth', 2);
        ylabel('I(L;A)');
        set(gca, 'XScale', 'log')
        hold on;
        
        subplot(2, 4, 5)
        plot(aIs - bIs, 'LineWidth', 2);
        ylabel('\deltaI(L;A)');
        set(gca, 'XScale', 'log')
        hold on;

        subplot(2, 4, 2)
        plot(Hs, 'LineWidth', 2);
        ylabel('H(L,A)');
        set(gca, 'XScale', 'log')
        hold on;
        
        subplot(2, 4, 6)
        plot(aHs - bHs, 'LineWidth', 2);
        ylabel('\deltaH(L,A)');
        set(gca, 'XScale', 'log')
        hold on;

        subplot(2, 4, 3)
        plot(Ds, 'LineWidth', 2);
        ylabel('D(L,A)');
        set(gca, 'XScale', 'log')
        hold on;
        
        subplot(2, 4, 4)
%         plot(aDs - bDs, 'LineWidth', 2);
%         ylabel('\deltaD(L,A)');
        plot(U, 'LineWidth', 2);
        ylabel('log(P(L|A))');
        set(gca, 'XScale', 'log')
        hold on;
        
        subplot(2, 4, 7)
        plot(diff(Ds), 'LineWidth', 2);
        ylabel('\DeltaD(L,A)');
        set(gca, 'XScale', 'log')
        hold on;
        
        subplot(2, 4, 8)
        plot(cumsum(diff(Ds), 'omitnan'), 'LineWidth', 2);
        ylabel('cumsum: \DeltaD(L,A) ');
        set(gca, 'XScale', 'log')
        hold on;
        
        drawnow
        
        if ~silent
            allbIs(o, :) = bIs;
            allbHs(o, :) = bHs;
            allbDs(o, :) = bDs;

            allaIs(o, :) = aIs;
            allaHs(o, :) = aHs;
            allaDs(o, :) = aDs;

            alldifDs(o, :) = diff(Ds);
        end
        
        toc
        close(f)
    end
    if ~silent
        data.allbIs = allbIs;
        data.allbHs = allbHs;
        data.allbDs = allbDs;

        data.allaIs = allaIs;
        data.allaHs = allaHs;
        data.allaDs = allaDs;

        data.alldifDs = alldifDs;
    else
        data = {};
    end
    
    if exist('setNames', 'var') && ~isempty(setNames)
        legend(setNames)
        data.setNames = setNames;
    end
end

function [P_ax, P_lx, P_LAx] = getProbs(teacher, learner)
    P_ax = teacher.P_ax;
    allYs = teacher.Ys_a;
    P_LAx = NaN(length(teacher.Xs), 1);
    P_lx = NaN(length(teacher.Xs), 1);
    for j = 1:length(teacher.Xs)
        if size(allYs(j, :, :), 2) == 1
            theseYs = squeeze(allYs(j, :, :))';
        else
            theseYs = squeeze(allYs(j, :, :));
        end
        [P_LAx(j), P_lx(j)] = getProbLAx(theseYs, learner);
    end
    return
end

function [P_LAx, P_lx] = getProbLAx(vars, learner)
    for i = 1:size(vars, 1)
        if ~exist('P_lx', 'var')||isempty(P_lx)
            [P_l, P_lx] = learner.getPx_ys(vars(i, :));
            P_LAx = P_l;
        else
            [P_l, ~] = learner.getPx_ys(vars(i, :), P_lx);
            P_LAx = P_LAx + P_l;
        end
    end
    P_LAx = P_LAx/size(vars, 1);
    return
end