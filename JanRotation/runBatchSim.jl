using PyPlot
using Distributions
using Random
using LinearAlgebra
using Statistics

struct StaticWorld
    #Sim Priors [position, object distance, object motion, velocity]
    A::Array{Float64,2}
    C::Array{Float64,2}
    muPrior::Array{Float64,2}
    endT::Float64
    dt::Float64
    endI::Int
end
StaticWorld(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,2}, endT::Float64, dt::Float64) =
    StaticWorld(A, C, muPrior, endT, dt, Integer(ceil(endT/dt)))
StaticWorld(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,2}) =
    StaticWorld(A, C, muPrior, 500., .5, 1000)

struct SimOpts
    sigmas::Array{Float64,1}
    a::Float64
    N::Int64
end
SimOpts(num::Int64) = SimOpts(ones(num,1))
SimOpts(static::StaticWorld) = SimOpts(ones(size(static.A,1)))
SimOpts(sigmas::Array{Float64,1}) = SimOpts(sigmas::Array{Float64,1}, 1, 500)

struct PlotOpts
    label::String
    color::Array{Float64,1}
    width::Float64
    alpha::Float64
end
PlotOpts(label::String) = PlotOpts(label::String, [0,0,1], 2, .3)
PlotOpts(label::String, color::Array{Float64,1}) = PlotOpts(label::String, color::Array{Float64,1}, 2, .3)

struct InitWorld
    sigmaIn::Array{Float64,2}
    noise
    Ks::Array{Array{Float64,2},1}
    Vars::Array{Array{Float64,2},1}
end
InitWorld(static::StaticWorld, simOpts::SimOpts) = InitWorld(static,
    diagm(simOpts.a*simOpts.sigmas),
    (static.C*(diagm(simOpts.sigmas))*static.C')/static.dt)
InitWorld(static::StaticWorld, initVar::Array{Float64,2}, sigmaIn::Array{Float64,2}) = InitWorld(
        sigmaIn, MvNormal(zeros(size(sigmaIn,1)), sigmaIn), getKalman(static, initVar, sigmaIn)...)

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
    a::Float64
    sigmas::Array{Float64,1}
    sigmaIn::Array{Float64,2}
    noise
    Ks::Array{Array{Float64,2},1}
    Vars::Array{Array{Float64,2},1}
    A::Array{Float64,2}
    C::Array{Float64,2}
    muPrior::Array{Float64,2}
    endT::Float64
    dt::Float64
    endI::Int128
    allT::Array{Float64,1}
    allAs::Array{Array{Float64,2},1}
end
FullWorld(static::StaticWorld, simOpts::SimOpts) =
    FullWorld(static::StaticWorld, InitWorld(static::StaticWorld, simOpts::SimOpts), simOpts.sigmas, simOpts.a)
FullWorld(static::StaticWorld, init::InitWorld, sigmas::Array{Float64,1}, a::Float64) =
    FullWorld(a, sigmas, init.sigmaIn, init.noise, init.Ks, init.Vars,
                static.A, static.C, static.muPrior, static.endT, static.dt, static.endI,
                Array{Float64,1}(0:static.dt:static.endI*static.dt),
                [static.A^t for t in 0:static.endI])

function runSim(kworld::FullWorld)
    Z = kworld.muPrior + vcat([sqrt(kworld.a*s)*randn(size(kworld.muPrior,2)) for s in kworld.sigmas]'...);
    Zs = [A*Z for A in kworld.allAs]
    Ys = [kworld.C*z + rand!(kworld.noise, similar(kworld.C*Z)) for z in Zs]

    Mus = Array{Array{Float64,2},1}(undef, kworld.endI+1)
    Mus[1] = kworld.muPrior
    for i = 2:kworld.endI+1
        Mus[i] = kworld.A*Mus[i-1] + kworld.Ks[i]*(Ys[i]-kworld.C*kworld.A*Mus[i-1]);
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

    rses = zeros(size(static.muPrior,1),static.endI+1,simOpts.N)*NaN
    for n = 1:simOpts.N#SHOULD BE PARFOR [if could start pool correctly...]
        rses[:,:,n] = runSim(kworld)
    end
    RMSE = dropdims(mean(rses, dims=3), dims=3)
    # sems = dropdims(std(rses; mean=RMSE,dims=3)/sqrt(simOpts.N), dims=3)
    eVars = dropdims(var(rses; mean=RMSE,dims=3), dims=3)
    mVars = hcat([diag(M) for M in kworld.Vars]...)
    plotRMSE(RMSE, eVars, mVars, kworld.allT, plotOpts)

    return rses
end
