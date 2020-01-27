function [data, Obs] = runDeterministicLeviathan(N, M, P, repeats)
    if nargin < 2
        M = [2, 7];
    end
    if nargin < 1
        N = 10;
    end
    Y = randperm(N);
    Z = randperm(N);
    Obs = cell(length(M)+1, length(Y)*length(Z));

    %% 1:1 pairing
    X = randperm(length(Y)*length(Z));
    k = 1;
    for i = 1:length(Y)
        for j = 1:length(Z)
            obs(k, :) = split(num2str([X(k), Y(i), Z(j)]))';
            k = k + 1;
        end
    end
    obs = obs(randperm(size(obs,1)), :);
    for r = 1:repeats
        obs = [obs; obs(randperm(size(obs,1)), :)];
    end
    Obs(1, 1:size(obs, 1), 1:size(obs,2)) = obs;
    labelStr{1} = '1:1';

    %% modulo M data structure
    Ind = 2;
    for m = M
%         P_a = zeros(m, length(Y), length(Z));
        X = cell2mat(arrayfun(@(y) mod(nansum([ones(size(Y))*y; Z], 1), m)+1, Y, 'UniformOutput', false));%mod(nansum([Y; Z], 1), m)+1;
        obs = cell(length(X), 3);
        k = 1;
        for i = 1:length(Y)
            for j = 1:length(Z)
%                 P_a(X(k), Y(i), Z(j)) = 1;
                obs(k, :) = split(num2str([X(k), Y(i), Z(j)]))';
                k = k + 1;
            end
        end
        
        obs = obs(randperm(size(obs,1)), :);
        for r = 1:repeats
            obs = [obs; obs(randperm(size(obs,1)), :)];
        end
        Obs(Ind, 1:size(obs, 1), 1:size(obs,2)) = obs;
        labelStr{Ind} = ['Mod' num2str(m)];
        Ind = Ind + 1;
    end
    
    %% Percentage distribution
    if nargin > 2
        for p = P'
            p = p(~isnan(p));
            p = p./sum(p);
            X = cell2mat(arrayfun(@(i) ones(1, round(p(i)*length(Y)*length(Z)))*i, 1:length(p), 'UniformOutput', false));
            while length(X) < length(Y)*length(Z)
                X{end + 1} = num2str(length(p));
            end
            obs = cell(length(X), 3);
            k = 1;
            for i = 1:length(Y)
                for j = 1:length(Z)
                    obs(k, :) = split(num2str([X(k), Y(i), Z(j)]))';
                    k = k + 1;
                end
            end
            obs = obs(randperm(size(obs,1)), :);
            for r = 1:repeats
                obs = [obs; obs(randperm(size(obs,1)), :)];
            end
            Obs(Ind, 1:size(obs, 1), 1:size(obs,2)) = obs;
            labelStr{Ind} = ['Prob[' num2str(p') ']'];
            Ind = Ind + 1;
        end
    end

    %% run it

    data = showLearning(Obs, labelStr, {}, false);%before (boolean), silent (boolean)
end