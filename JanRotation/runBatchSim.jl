using PyPlot
using Distributions
using Random
using LinearAlgebra
using Statistics
using Distributed

struct StaticWorld
    #Sim Priors [position, object distance, object motion, velocity]
    A::Array{Float64,2}
    C::Array{Float64,2}
    muPrior::Array{Float64,2}
    endT::Float64
    dt::Float64
    endI::Int
    allT::Array{Float64,1}
    allAs::Array{Array{Float64,2},1}
end
StaticWorld(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,2}, endT::Float64, dt::Float64) =
    StaticWorld(A, C, muPrior, endT, dt, Integer(ceil(endT/dt)))
StaticWorld(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,2}) =
    StaticWorld(A, C, muPrior, 500., .5, 1000)
StaticWorld(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,2}, endT::Float64, dt::Float64, endI::Int) =
    StaticWorld(A, C, muPrior, endT, dt, endI,
                Array{Float64,1}(0:dt:endI*dt),
                [A^t for t in 0:endI])

struct SimOpts
    sigmas::Array{Float64,1}
    a::Float64
    N::Int64
end
SimOpts(num::Int64) = SimOpts(ones(num,1))
SimOpts(static::StaticWorld) = SimOpts(ones(size(static.A,1)))
SimOpts(sigmas::Array{Float64,1}) = SimOpts(sigmas, 1, 500)

struct PlotOpts
    label::String
    color::Array{Float64,1}
    width::Float64
    alpha::Float64
end
PlotOpts(label::String) = PlotOpts(label, [0,0,1], 2, .2)
PlotOpts(label::String, color::Array{Float64,1}) = PlotOpts(label, color, 2, .2)

function getKalman(static::StaticWorld, initVar::Array{Float64,2}, sigmaIn::Array{Float64,2})
    Vars = Array{Array{Float64,2}, 1}(undef, static.endI+1)
    Ks = Array{Array{Float64,2}, 1}(undef, static.endI+1)
    Vars[1] = initVar
    #Get filter
    for i = 2:static.endI+1
        Ks[i] = (static.A*Vars[i-1]*static.A'*static.C')/(static.C*static.A*Vars[i-1]*static.A'*static.C' + sigmaIn)
        Vars[i] = (I - Ks[i]*static.C)*static.A*Vars[i-1]*static.A'
    end
    return Ks, Vars
end

struct FullWorld
    static::StaticWorld
    a::Float64
    sigmaIn::Array{Float64,2}
    noise
    Ks::Array{Array{Float64,2},1}
    Vars::Array{Array{Float64,2},1}
end
FullWorld(static::StaticWorld, simOpts::SimOpts) =
    FullWorld(static::StaticWorld, simOpts.a,
    diagm(simOpts.a*simOpts.sigmas), (static.C*diagm(simOpts.sigmas)*static.C')/static.dt)
FullWorld(static::StaticWorld, a::Float64, initVar::Array{Float64,2}, sigmaIn::Array{Float64,2}) =
    FullWorld(static, a, sigmaIn, MvNormal(zeros(size(sigmaIn,1)), sigmaIn), getKalman(static, initVar, sigmaIn)...)

@everywhere function runSim(kworld::FullWorld)
    Z = kworld.static.muPrior + vcat([sqrt(kworld.a*s)*randn(size(kworld.static.muPrior,2)) for s in diag(kworld.Vars[1])]'...);
    Zs = [A*Z for A in kworld.static.allAs]
    Ys = [kworld.static.C*z + rand!(kworld.noise, similar(kworld.static.C*Z)) for z in Zs]

    Mus = Array{Array{Float64,2},1}(undef, kworld.static.endI+1)
    Mus[1] = kworld.static.muPrior
    for i = 2:kworld.static.endI+1
        Mus[i] = kworld.static.A*Mus[i-1] + kworld.Ks[i]*(Ys[i]-kworld.static.C*kworld.static.A*Mus[i-1]);
    end
    RSE = broadcast(rse, Mus - Zs)#convert to distance from components
    return hcat(RSE...)
end

function rse(M::Array{Float64,2})
    return dropdims(sqrt.(sum(M.^2,dims=2)), dims=2)
end

function plotshade(y::Array{Float64,1}, err::Array{Float64,1}, x::Array{Float64,1}, opts::PlotOpts)
    f = fill_between(x[1:length(y)], y+err, y-err,color=opts.color, alpha=opts.alpha, linestyle="-.", hatch="/")
    p = plot(x[1:length(y)],y,color=opts.color[:],linewidth = opts.width, label=opts.label) ## change color | linewidth to adjust mean line()
end

function plotRMSE(RMSE::Array{Float64,2}, eVars::Array{Float64,2}, mVars::Array{Float64,2}, allT::Array{Float64,1}, opts::PlotOpts)

    subplot(2, 2, 1)
    plotshade(RMSE[1,:], eVars[1,:], allT, opts)
    fill_between(allT, RMSE[1,:]+mVars[1,:], RMSE[1,:]-mVars[1,:], color=opts.color, alpha=opts.alpha, linestyle=":", hatch="|")
    xlabel("time")
    ylabel("root mean square error")
    title("Position RMSE")

    subplot(2, 2, 2)
    plotshade(RMSE[2,:], eVars[2,:], allT, opts)
    fill_between(allT, RMSE[2,:]+mVars[2,:], RMSE[2,:]-mVars[2,:], color=opts.color, alpha=opts.alpha, linestyle=":", hatch="|")
    xlabel("time")
    ylabel("root mean square error")
    title("Object Distance RMSE")

    subplot(2, 2, 3)
    plotshade(RMSE[4,:], eVars[4,:], allT, opts)
    fill_between(allT, RMSE[4,:]+mVars[4,:], RMSE[4,:]-mVars[4,:], color=opts.color, alpha=opts.alpha, linestyle=":", hatch="|")
    xlabel("time")
    ylabel("root mean square error")
    title("Velocity RMSE")

    return
end

function runBatchSim(plotOpts::PlotOpts, static::StaticWorld, simOpts::SimOpts)
    kworld = FullWorld(static, simOpts)#SLOW

    rses = Array{Array{Float64,2},1}(undef, simOpts.N)
    pmap(n->rses[n] = runSim(kworld), 1:simOpts.N)
    # for n = 1:simOpts.N#SHOULD BE PARFOR [if could start pool correctly...]
    #     rses[:,:,n] = runSim(kworld)
    # end
    RMSE = mean(rses)
    # sems = dropdims(std(rses; mean=RMSE,dims=3)/sqrt(simOpts.N), dims=3)
    eVars = var(rses; mean=RMSE)
    mVars = hcat([diag(M) for M in kworld.Vars]...)
    plotRMSE(mean(rses), eVars, mVars, kworld.static.allT, plotOpts)

    return rses
end
