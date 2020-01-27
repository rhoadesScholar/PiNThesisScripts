X = 0;
V = 10;
dt = .01;
sigVest = 2;
a = 1;
endT = 9999;

C = [0 1];
A = [1 dt; 0 1];

vVest = gmdistribution(V, sigVest);

oldVar = eye(size(A))*a*sigVest;
muInit = [X; V];
oldMu = muInit;
oldZ = muInit;

K = @(oldVar) (A*oldVar*C'* (C*oldVar*C' + sigVest/dt)^-1);
Var = @(oldVar, K) (eye(size(A))-K(oldVar)*C)*A*oldVar*A';
Mu = @(oldMu, oldVar, K, vIn_t) A*oldMu + K(oldVar)*(vIn_t - C*A*oldMu);
Z = @(oldZ) A*oldZ;

calcVar = @(t) (sigVest/(t + (1/a)))* [(t^2 + a*t + 1) t; t 1];
calcMu = @(t, vIn, j) [1 t(j)*(1 - t(j)/(t(j) + (1/a))); 0 (1 - t(j)/(t(j) + (1/a)))]*muInit + ...
    sum(cell2mat(arrayfun(@(v) v*(dt/(t(j) + (1/a)))*[t(j); 1], vIn(1:j)',  'UniformOutput', false)), 2);

t = dt:dt:endT;
vIn = zeros(length(t), 1);

fig = figure('DeleteFcn', 'evalin(''caller'', ''done = true;'')');
done = false; 

subplot(2, 2, 1)
varLine = plot(0, 0, 'LineWidth', 2, 'Color', 'r');
hold on
muLine = plot(0, 0, 'LineWidth', 2, 'Color', 'b');
legend('\Sigma Difference', '\mu Difference');
title('Method Differences')

subplot(2, 2, 3)
stepXLine = plot(0, 0, 'LineWidth', 2, 'Color', 'r');
hold on
calcXLine = plot(0, 0, 'LineWidth', 2, 'Color', 'b');
legend('Step', 'Closed form');
title('Percent Error: Position')

subplot(2, 2, 4)
stepVLine = plot(0, 0, 'LineWidth', 2, 'Color', 'r');
hold on
calcVLine = plot(0, 0, 'LineWidth', 2, 'Color', 'b');
% legend('Step', 'Closed form');
title('Percent Error: Velocity')

for j = 1:length(t)
    vIn(j) = random(vVest);
    
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
     
    stepXDiff = (oldMu(1) - oldZ(1))/oldZ(1); 
    calcXDiff = (thisCalcMu(1) - oldZ(1))/oldZ(1);
    stepXLine.YData(end+1) = stepXDiff;
    stepXLine.XData(end+1) = t(j);
    calcXLine.YData(end+1) = calcXDiff;
    calcXLine.XData(end+1) = t(j);
    
    stepVDiff = (oldMu(2) - oldZ(2))/oldZ(2);
    calcVDiff = (thisCalcMu(2) - oldZ(2))/oldZ(2);
    stepVLine.YData(end+1) = stepVDiff;
    stepVLine.XData(end+1) = t(j);
    calcVLine.YData(end+1) = calcVDiff;
    calcVLine.XData(end+1) = t(j);
    
    drawnow
    if done || strcmp(fig.CurrentCharacter, ' '), break; end
end
