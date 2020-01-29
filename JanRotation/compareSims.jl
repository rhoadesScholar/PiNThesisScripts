include("runBatchSim.jl")
include("linspecer.jl")
using Printf

dt = .1
A = [1 0 0 dt;
     0 1 dt -dt;
     0 0 1 0;
     0 0 0 1]
C = [0. 1. 0 0;
     0 0 0 1]
muPrior = [0. 0.;
           10. 10.;
           0. 0.;
           1. 0.]
endT = 100
sigmas = [1., 100, 50, 1]

static = StaticWorld(A, C, muPrior, endT, dt)
simBase = SimOpts(sigmas)

flexSigValues = [.01, .1, 1, 10, 100]
flexSigInd = 4
flexVarName = "sigmaVest"

function compareSims(static::StaticWorld, simBase::SimOpts, flexSigValues::Array{Float64,1}, flexSigInd::Int64, flexVarName::String)
    variationNum = length(flexSigValues)
    colors = linspecer(variationNum)
    figure()
    for v = 1:variationNum
        println(string("Variation #", v))
        plotOpts = PlotOpts(@sprintf("%s = %s", flexVarName, string.(flexSigValues[v])), colors[v,:])
        simOpts = SimOpts(simBase, flexSigInd, flexSigValues[v])
        runBatchSim(plotOpts, static, simOpts)
    end

    legend()
end