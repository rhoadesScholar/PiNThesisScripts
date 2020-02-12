include("runBatchCompareSim.jl")
include("linspecer.jl")
using Printf

function compareSims(As::Array{Array{Float64,2},1}, C::Array{Float64,2}, muPrior::Array{Float64,1}, endT::Number, dt::Number,
                    sigmas::Array{Float64,1}, simNames::Array{String,1})
    set_zero_subnormals(true)
    variationNum = length(As)
    colors = linspecer(variationNum)
    # figure()
    for v = 1:variationNum
        println(string("Variation #", v))
        plotOpts = PlotOpts(simNames[v], colors[v,:])
        simOpts = SimOpts(sigmas, 1000)
        static = StaticWorld(As[v], C, muPrior, endT, dt)
        @time runBatchSim(plotOpts, static, simOpts)
    end
    legend()

    return
end

function compareSims(As::Array{Array{Float64,2},1}, C::Array{Float64,2}, muPrior::Array{Float64,1}, sigmas::Array{Float64,1}, simNames::Array{String,1})
    set_zero_subnormals(true)
    variationNum = length(As)
    colors = linspecer(variationNum)
    # figure()
    for v = 1:variationNum
        println(string("Variation #", v))
        plotOpts = PlotOpts(simNames[v], colors[v,:])
        simOpts = SimOpts(sigmas, 1000)
        static = StaticWorld(As[v], C, muPrior)
        @time runBatchSim(plotOpts, static, simOpts)
    end
    legend()

    return
end
