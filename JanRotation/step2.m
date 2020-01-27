X = 0;
V = 10;
dt = .1;
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
calcMu = @(t, vIn, j) [1 dt*(1 - t/(t+(1/a))); 0 (1 - t/(t + (1/a)))]*muInit + ...
    sum(cell2mat(arrayfun(@(v, i) v*(dt/(i*dt + (1/a)))*[i*dt; 1], vIn(1:j)', 1:j,  'UniformOutput', false)), 2);

t = dt:dt:endT;
vIn = zeros(length(t), 1);

varLine = plot(0, 0, 'LineWidth', 2, 'Color', 'r');
hold on
muLine = plot(0, 0, 'LineWidth', 2, 'Color', 'b');
for j = 1:length(t)
    vIn(j) = random(vVest);
    
    oldVar = Var(oldVar, K);
    varDiff = sum(abs(oldVar - calcVar(t(j))), 'all');%./(oldVar + calcVar(t))/2
    varLine.YData(end+1) = varDiff;
    varLine.XData(end+1) = t(j);
    
    oldMu = Mu(oldMu, oldVar, K, vIn(j));
    muDiff = sum(abs(oldMu - calcMu(t(j), vIn, j)), 'all');%./(oldMu + calcMu(t, vIn))/2
    muLine.YData(end+1) = muDiff;
    muLine.XData(end+1) = t(j);
    
    drawnow
end
