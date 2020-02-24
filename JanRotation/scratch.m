dt = .1;
A1 = [1. 0 0 0 0 0 dt 0;
      0 1 0 0 0 0 0 dt;
      0 0 1 0 dt 0 -dt 0;
      0 0 0 1 0 dt 0 -dt;
      0 0 0 0 1 0 0 0;
      0 0 0 0 0 1 0 0;
      0 0 0 0 0 0 1 0;
      0 0 0 0 0 0 0 1];

 A2 = [1. 0 0 0 0 0 dt 0;
       0 1 0 0 0 0 0 dt;
       0 0 1 0 0 0 -dt 0;
       0 0 0 1 0 0 0 -dt;
       0 0 0 0 1 0 0 0;
       0 0 0 0 0 1 0 0;
       0 0 0 0 0 0 1 0;
       0 0 0 0 0 0 0 1];

As{1} = A1;
As{2} = A2;

C = [0. 0 1 0 0 0 0 0;
     0. 0 0 1 0 0 0 0;
     0. 0 0 0 0 0 1 0;
     0. 0 0 0 0 0 0 1];
muPrior = [0., 0, 10., 10, 0., 1, 1., 0]';
endT = 100.;
initVar = [1., 1, 10., 10, 10., 10, 100., 100]';
emitVar = [1., 1, 10., 10, 10., 10, 100., 100]';
a = 1.;

for i = 1:length(As)
      KMs(i) = KalmanModel(As{i}, C, muPrior, initVar, a, ceil(endT/dt)+1);
      SWs(i) = SimWorld(As{i}, C, muPrior, emitVar, endT, dt);
end

N=100;

E = .1;
Vars = cat(4, KMs.Vars);
Sigma = @(sigmas) reshape(cell2mat(arrayfun(@(i) nearestSPD(sigmas(:,:,i)), 1:size(sigmas,3), 'UniformOutput', false)), size(sigmas,1), size(sigmas,2), []);

MusLL = NaN(length(SWs), length(KMs), length(muPrior)+1, ceil(endT/dt)+1);
for s = 1:length(SWs)
    SEs = NaN(length(KMs), N, length(muPrior)+1, ceil(endT/dt)+1);
    eYs = NaN(length(KMs), size(C,1), ceil(endT/dt)+1);
    Mus = NaN(length(KMs), length(muPrior)+1, ceil(endT/dt)+1);
    for i = 1:N
        [Zs, Ys] = SWs(s).getStates();
        for k = 1:length(KMs)
            [SEs(k,i,:,:), eYs(k,:,:), Mus(k,:,:)] =  KMs(k).runSim(Zs, Ys);
        end
%         ps = softmax(squeeze(SEs(:,i,end,:)));
%         
%         dists = arrayfun(@(t) gmdistribution(Mus(:,1:end-1,t), Sigma(squeeze(Vars(:,:,t,:))), ps(:,t)), 1:size(Vars,3), 'UniformOutput', false);
%         R = @(X, E, dist) diff(cdf(dist, [X-E, X+E]));
%         getMetaMus = @(D, E) solve(R(X, E, D) == max(R(Y, E, D)), X);
%         cellfun(@(D) getMetaMus(D, E), dists)
        

    end
    MusLL(s,:,:,:) =  squeeze(nanmean(SEs, 2));
end

dims = 2;

labels = {'Moving', 'Still'; 'Moving', 'Still'};

plotMSE(MusLL, dims, Vars, SWs(1).allT, labels)






findmax(maxval, ind, A)


static = StaticWorld(A, C, muPrior, endT, dt);

flexSigValues = [10, 1, .1];
flexSigInd = 4;
flexVarName = "sigmaVest";

v=1;

simNames{1} = "Moving";
simNames{2} = "Still";

set_zero_subnormals(true)
variationNum = length(simNames);
colors = linspecer(variationNum);

simOpts = SimOpts(sigmas, 1000);
plotOpts = PlotOpts(simNames(v), colors(v,:))
kworld = FullWorld(static, simOpts);
