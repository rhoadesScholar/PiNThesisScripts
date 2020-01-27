function compareSims(flexVarName, flexVarValues)
    staticVarStr = '';
    
    if ~exist('flexVarName', 'var') || isempty(flexVarName)
        flexVarName = 'sigmaVest';
    end
    if ~exist('flexVarValues', 'var') || isempty(flexVarValues)
        flexVarValues = {.01 .1 1 10 100};%must be cell array of arrays
    end
    variationNum = numel(flexVarValues);
    colors = linspecer(variationNum);
    figure
    for v = 1:variationNum
        thisFlexArg = sprintf('%s=%s;', flexVarName, mat2str(flexVarValues{v}));
        argStr = [staticVarStr, '; ', thisFlexArg, sprintf('color=%s;',mat2str(colors(v,:)))];
        runBatchSim(argStr);
    end
    legend(cellfun(@(b) [flexVarName ' = ' mat2str(b)], flexVarValues, 'UniformOutput', false))
end