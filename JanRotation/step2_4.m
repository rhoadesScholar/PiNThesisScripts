X = 0;
V = 10;
dt = .001;
sigVest = 2;
a = 1;
endT = 100;

C = [0 1];
A = [1 dt; 0 1];

oldVar = eye(size(A))*a*sigVest;
oldZ = [X; V] + sqrt(a*sigVest)*[randn(); randn()];
muInit = [X; V];
oldMu = muInit;
vVest = gmdistribution(oldZ(2), sigVest);

K = @(oldVar) (A*oldVar*C'* (C*oldVar*C' + sigVest/dt)^-1);
Var = @(oldVar, K) (eye(size(A))-K(oldVar)*C)*A*oldVar*A';
Mu = @(oldMu, oldVar, K, vIn_t) A*oldMu + K(oldVar)*(vIn_t - C*A*oldMu);
Z = @(oldZ) A*oldZ;

calcVar = @(t) (sigVest/(t + (1/a)))* [(t^2 + a*t + 1) t; t 1];
calcMu = @(t, vIn, j) [1 t(j)*(1 - t(j)/(t(j) + (1/a))); 0 (1 - t(j)/(t(j) + (1/a)))]*muInit + ...
    sum(vIn(1:j))*(dt/(t(j) + (1/a)))*[t(j); 1];

t = dt:dt:endT;
vIn = random(vVest, length(t));

fig = figure('DeleteFcn', 'evalin(''caller'', ''done = true;'')');
done = false; 

subplot(2, 2, 1)
varLine = plot(NaN, NaN, 'LineWidth', 2, 'Color', 'r');
hold on
muLine = plot(NaN, NaN, 'LineWidth', 2, 'Color', 'b');
legend('\Sigma Difference', '\mu Difference');
title('Method Differences')

subplot(2, 2, 3)
stepXLine = plot(NaN, NaN, 'LineWidth', 2, 'Color', 'r');
hold on
calcXLine = plot(NaN, NaN, 'LineWidth', 2, 'Color', 'b');
trueXLine = plot(NaN, NaN, 'LineWidth', 2, 'Color', 'g');
legend('Step', 'Closed form', 'Actual');
title('Position')

subplot(2, 2, 4)
stepVLine = plot(NaN, NaN, 'LineWidth', 2, 'Color', 'r');
hold on
calcVLine = plot(NaN, NaN, 'LineWidth', 2, 'Color', 'b');
trueVLine = plot(NaN, NaN, 'LineWidth', 2, 'Color', 'g');
% legend('Step', 'Closed form');
title('Velocity')

for j = 1:length(t)
    
%     %$$$$acceleration?
%     vIn(j) = vIn(j) + .5*t(j);
    
    thisCalcMu = calcMu(t, vIn, j);
    
    oldVar = Var(oldVar, K);
    varDiff = sum(abs(oldVar - calcVar(t(j))), 'all');
    varLine.YData(end+1) = varDiff;
    varLine.XData(end+1) = t(j);
    
    oldMu = Mu(oldMu, oldVar, K, vIn(j));
    muDiff = sum(abs(oldMu - thisCalcMu), 'all');
    muLine.YData(end+1) = muDiff;
    muLine.XData(end+1) = t(j);
    
    oldZ = Z(oldZ);
    
    stepXLine.YData(end+1) = oldMu(1);
    stepXLine.XData(end+1) = t(j);
    calcXLine.YData(end+1) = thisCalcMu(1);
    calcXLine.XData(end+1) = t(j);
    trueXLine.YData(end+1) = oldZ(1);
    trueXLine.XData(end+1) = t(j);
    
    stepVLine.YData(end+1) = oldMu(2);
    stepVLine.XData(end+1) = t(j);
    calcVLine.YData(end+1) = thisCalcMu(2);
    calcVLine.XData(end+1) = t(j);
    trueVLine.YData(end+1) = oldZ(2);
    trueVLine.XData(end+1) = t(j);
    
    if done || strcmp(fig.CurrentCharacter, ' '), break; end
end
