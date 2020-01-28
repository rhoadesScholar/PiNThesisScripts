include("runBatchSim.jl")
include("linspecer.jl")
using Printf

function compareSims(flexVarName, flexVarValues)
    staticVarStr = Meta.parse("")

    variationNum = length(flexVarValues)
    colors = linspecer(variationNum)
    figure()
    for v = 1:variationNum
        thisFlexArg = Meta.parse(@sprintf("%s=%s", flexVarName, string.(flexVarValues[v])))
        colorArg = :(color = $(colors[v,:]))
        labelArg = Meta.parse(@sprintf("label = \"%s = %s\"", flexVarName, string.(flexVarValues[v])))
        argS = [thisFlexArg colorArg labelArg]
        # println(argS)
        runBatchSim(argS)
    end

    legend()
end
