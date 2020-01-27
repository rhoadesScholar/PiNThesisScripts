X = 0;
V = 10;
dt = .1;
sigVest = 2;
a = 1;
endT = 100;
%use random draw for initial

C = [0 1];
A = [1 dt; 0 1];

vVest = gmdistribution(V, sigVest);

t = dt:dt:endT;

oldVar = zeros(numel(C), numel(C), length(t));
oldVar(:,:,1) = eye(size(A))*a*sigVest;
oldZ = zeros(numel(C), length(t));
oldZ(:, 1) = [X; V];
muInit = oldZ + sqrt(sigVest)*[randn(); randn()];
oldMu = zeros(numel(C), length(t));
oldMu(:,1) = muInit;
thisCalcVar = zeros(numel(C), numel(C), length(t));
thisCalcVar(:,:,1) = eye(size(A))*a*sigVest;

K = @(oldVar) (A*oldVar*C'* (C*oldVar*C' + sigVest/dt)^-1);
Var = @(oldVar, K) (eye(size(A))-K(oldVar)*C)*A*oldVar*A';
Mu = @(oldMu, oldVar, K, vIn_t) A*oldMu + K(oldVar)*(vIn_t - C*A*oldMu);
Z = @(oldZ) A*oldZ;

calcVar = @(t) (sigVest/(t + (1/a)))* [(t^2 + a*t + 1) t; t 1];
calcMu = @(t, vIn, i) cell2mat(arrayfun(@(j) [1 t(j)*(1 - t(j)/(t(j) + (1/a))); 0 (1 - t(j)/(t(j) + (1/a)))]*muInit + ...
    sum(cell2mat(arrayfun(@(v) v*(dt/(t(j) + (1/a)))*[t(j); 1], vIn(1:j)',  'UniformOutput', false)), 2), 1:i,  'UniformOutput', false));

vIn = random(vVest,length(t));
    
for j = 2:length(t)
    oldVar(:,:, j) = Var(oldVar(:,:,j-1), K);
    oldMu(:, j) = Mu(oldMu(:,j-1), oldVar(:,:,j-1), K, vIn(j));
    oldZ(:, j) = Z(oldZ(:, j-1));
    thisCalcVar(:,:,j) = calcVar(t(j));
end

thisCalcMu = calcMu(t, vIn, j);

varDiff = sum(abs(oldVar - thisCalcVar), 'all');
muDiff = sum(abs(oldMu - thisCalcMu), 'all');


stepXDiff = (oldMu(1,:) - oldZ(1,:))/oldZ(1,:); 
calcXDiff = (thisCalcMu(1,:) - oldZ(1,:))/oldZ(1,:);

stepVDiff = (oldMu(2,:) - oldZ(2,:))/oldZ(2,:);
calcVDiff = (thisCalcMu(2,:) - oldZ(2,:))/oldZ(2,:);

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