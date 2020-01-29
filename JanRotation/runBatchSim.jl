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
    endT::Number
    dt::Number
    endI::Int
end
StaticWorld(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,2}, endT::Number, dt::Number) =
    StaticWorld(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,2}, endT::Number, dt::Number, Integer(ceil(endT/dt)))
StaticWorld(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,2}) =
    StaticWorld(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,2}, 500, .5, 1000)

struct SimOpts
    sigmas::Array{Float64,1}
    a::Float64
    N::Int64
end
SimOpts(num::Int64) = SimOpts(ones(num,1))
SimOpts(static::StaticWorld) = SimOpts(ones(size(static.A,1)))
SimOpts(sigmas::Array{Float64,1}) = SimOpts(sigmas::Array{Float64,1}, 1, 1000)
SimOpts(base::SimOpts, ind::Int64, sig::Number) = SimOpts(setindex!(base.sigmas, sig, ind), base.a, base.N)

struct PlotOpts
    label::String
    color::Array{Float64,1}
    width::Float64
    alpha::Float64
end
PlotOpts(label::String) = PlotOpts(label::String, [0,0,1], 2, .3)
PlotOpts(label::String, color::Array{Float64,1}) = PlotOpts(label::String, color::Array{Float64,1}, 2, .3)

struct InitWorld
    Z::Array{Float64,2}
    Var::Array{Float64,2}
    sigmaPrior::Array{Float64,2}
end
InitWorld(static::StaticWorld, simOpts::SimOpts) = InitWorld(
    static.muPrior + vcat([sqrt(simOpts.a*s)*randn(size(static.muPrior,2)) for s in simOpts.sigmas]'...),
    diagm(simOpts.a*simOpts.sigmas),
    static.C*(diagm(simOpts.a*simOpts.sigmas))*static.C'/static.dt)

struct FullWorld
    Zs::Array{Array{Float64,2},1}
    initVar::Array{Float64,2}
    sigmaPrior::Array{Float64,2}
    noise
    A::Array{Float64,2}
    C::Array{Float64,2}
    muPrior::Array{Float64,2}
    endT::Float64
    dt::Float64
    endI::Int128
end
FullWorld(static::StaticWorld, init::InitWorld) =
    FullWorld([(static.A^t)*init.Z for t in 0:static.endI], init.Var, init.sigmaPrior,
    MvNormal(zeros(size(init.sigmaPrior,1)), init.sigmaPrior),
    static.A, static.C, static.muPrior, static.endT, static.dt, static.endI)
FullWorld(static::StaticWorld, simOpts::SimOpts) =
    FullWorld(static::StaticWorld, InitWorld(static::StaticWorld, simOpts::SimOpts))

function runSim(kworld::FullWorld)
    err = zeros(size(kworld.muPrior,1),kworld.endI+1)*NaN
    oldVar = kworld.initVar
    oldMu = kworld.muPrior
    err[:,1] = sum((oldMu - kworld.Zs[1]).^2,dims=2)
    for i = 2:kworld.endI+1
        #Get observations
        Y = C*kworld.Zs[i] + rand!(kworld.noise,zeros(size(C*kworld.Zs[i])))

        #Get filter()
        K = kworld.A*oldVar*kworld.A'*kworld.C'/(kworld.C*kworld.A*oldVar*kworld.A'*kworld.C' + kworld.sigmaPrior)
        oldVar = (I - K*kworld.C)*kworld.A*oldVar*kworld.A'
        oldMu = kworld.A*oldMu + K*(Y-kworld.C*kworld.A*oldMu);
        err[:,i] = sum((oldMu - kworld.Zs[i]).^2,dims=2)
    end
    return err
end

function plotshade(y, err, x, opts::PlotOpts)
    f = fill_between(x[1:length(y)], y+err, y-err,color=opts.color, alpha=opts.alpha)

    # f.Annotation.LegendInformation.IconDisplayStyle = "off"
    p = plot(x[1:length(y)],y,color=opts.color[:],linewidth = opts.width, label=opts.label) ## change color | linewidth to adjust mean line()
    # @show p
end

function plotRMSE(RMSE, sems, dt, opts::PlotOpts)
    endI = size(RMSE,2)-1

    subplot(2, 2, 1)
    plotshade(RMSE[1,:], sems[1,:], 0:dt:endI*dt, opts)
    xlabel("time")
    ylabel("root mean square error")
    title("Position RMSE")

    subplot(2, 2, 3)
    plotshade(RMSE[2,:], sems[2,:], 0:dt:endI*dt, opts)
    xlabel("time")
    ylabel("root mean square error")
    title("Object Distance RMSE")

    subplot(2, 2, 4)
    plotshade(RMSE[4,:], sems[4,:], 0:dt:endI*dt, opts)
    xlabel("time")
    ylabel("root mean square error")
    title("Velocity RMSE")
    return
end

function runBatchSim(plotOpts::PlotOpts, static::StaticWorld, simOpts::SimOpts)
    kworld = FullWorld(static, simOpts)

    errs = zeros(size(static.muPrior,1),static.endI+1,simOpts.N)*NaN
    for n = 1:simOpts.N#SHOULD BE PARFOR [if could start pool correctly...]
        errs[:,:,n] = runSim(kworld)
    end
    RMSE = sqrt.(dropdims(mean(errs, dims=3), dims=3))
    sems = dropdims(std(errs;mean=RMSE,dims=3)/sqrt(simOpts.N), dims=3)
    plotRMSE(RMSE, sems, static.dt, plotOpts)

    return errs
end
