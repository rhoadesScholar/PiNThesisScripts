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
muPrior = [0., 0, 10., 0, 0., 1, 1., 0]';
endT = 1000.;
initVar = [1., 1, 100., 100, 1., 1, 1., 1]';
emitVar = [1., 1, 100., 100, 1., 1, 1., 1]';
a = 1.;

for i = 1:length(As)
      KMs(i) = KalmanModel(As{i}, C, muPrior, initVar, a, ceil(endT/dt)+1);
      SWs(i) = SimWorld(As{i}, C, muPrior, emitVar, endT, dt);
end

N=500;

MusLL = NaN(length(SWs), length(KMs), length(muPrior)+1, ceil(endT/dt)+1);
for s = 1:length(SWs)
    temp = NaN(length(KMs), N, length(muPrior)+1, ceil(endT/dt)+1);
    for i = 1:N
        [Zs, Ys] = SWs(s).getStates();
        for k = 1:length(KMs)
            temp(k,i,:,:) =  KMs(k).runSim(Zs, Ys);
        end
    end
    MusLL(s,:,:,:) =  squeeze(nanmean(temp, 2));
end

Vars = cat(4, KMs(:).Vars);
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
