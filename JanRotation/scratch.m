dt = .2;
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

epsilon = .1;
Vars = cat(4, KMs.Vars);

Bond007 = Agent(KMs, epsilon, SWs);

MusLL = NaN(length(SWs), length(KMs)+1, length(muPrior)+1, ceil(endT/dt)+1);
for s = 1:length(SWs)
    SEs = NaN(length(KMs)+1, N, length(muPrior)+1, ceil(endT/dt)+1);
    eYs = NaN(length(KMs), size(C,1), ceil(endT/dt)+1);
    Mus = NaN(length(KMs), length(muPrior)+1, ceil(endT/dt)+1);
    for i = 1:N
        [Zs, Ys] = SWs(s).getStates();
        for k = 1:length(KMs)
            [SEs(k,i,:,:), eYs(k,:,:), Mus(k,:,:)] =  KMs(k).runSim(Zs, Ys);
        end
        
        SEs(k+1,i,:,:) = Bond007.getMetaMus(Mus, Zs, Ys);        

    end
    MusLL(s,:,:,:) =  squeeze(nanmean(SEs, 2));
end

dims = 2;

labels = {'Moving', 'Still'; 'Moving', 'Still', 'Combined'};

plotMSE(MusLL, dims, Vars, SWs(1).allT, labels)