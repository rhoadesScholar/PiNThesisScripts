function data = runDeterministicLeviathan(N, M)
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
    % obs(:, 1) = split(num2str(Y));
    % obs(:, 1) = split(num2str(Z));
    Obs(1, 1:size(obs, 1), 1:size(obs,2)) = obs;
    labelStr{1} = '1:1';

    %% modulo M data structure
    mInd = 2;
    for m = M
        P_a = zeros(m, length(Y), length(Z));
        X = cell2mat(arrayfun(@(y) mod(nansum([ones(size(Y))*y; Z], 1), m)+1, Y, 'UniformOutput', false));%mod(nansum([Y; Z], 1), m)+1;
        obs = cell(length(X), 3);
        k = 1;
        for i = 1:length(Y)
            for j = 1:length(Z)
                P_a(X(k), Y(i), Z(j)) = 1;
                obs(k, :) = split(num2str([X(k), Y(i), Z(j)]))';
                k = k + 1;
            end
        end
        Obs(mInd, 1:size(obs, 1), 1:size(obs,2)) = obs;
        labelStr{mInd} = ['Mod' num2str(m)];
        mInd = mInd + 1;
    end

    %% run it

    data = showLearning(Obs, labelStr, true);
end
