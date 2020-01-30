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
end
FullWorld(static::StaticWorld, simOpts::SimOpts) =
    FullWorld(static::StaticWorld, InitWorld(static::StaticWorld, simOpts::SimOpts), simOpts.sigmas, simOpts.a)
FullWorld(static::StaticWorld, init::InitWorld, sigmas::Array{Float64,1}, a::Float64) =
    FullWorld(a, sigmas, init.sigmaIn, init.noise, init.Ks, init.Vars, static.A, static.C, static.muPrior, static.endT, static.dt, static.endI)

function runSim(kworld::FullWorld)
    Z = kworld.muPrior + vcat([sqrt(kworld.a*s)*randn(size(kworld.muPrior,2)) for s in kworld.sigmas]'...);
    Zs = [(kworld.A^t)*Z for t in 0:kworld.endI]
    Ys = [kworld.C*z + rand!(kworld.noise, zeros(size(kworld.C*Z))) for z in Zs]

    err = zeros(size(kworld.muPrior,1),kworld.endI+1)*NaN
    oldMu = kworld.muPrior
    err[:,1] = sum((oldMu - Zs[1]).^2,dims=2)
    for i = 2:kworld.endI+1
        #Get observations
        # Y = kworld.C*kworld.Zs[i] + rand!(kworld.noise, zeros(size(kworld.C*kworld.Zs[i])))
        oldMu = kworld.A*oldMu + kworld.Ks[i]*(Ys[i]-kworld.C*kworld.A*oldMu);
        err[:,i] = sum((oldMu - Zs[i]).^2,dims=2)
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

    subplot(2, 2, 2)
    plotshade(RMSE[2,:], sems[2,:], 0:dt:endI*dt, opts)
    xlabel("time")
    ylabel("root mean square error")
    title("Object Distance RMSE")

    subplot(2, 2, 3)
    plotshade(RMSE[4,:], sems[4,:], 0:dt:endI*dt, opts)
    xlabel("time")
    ylabel("root mean square error")
    title("Velocity RMSE")

    # subplot(2, 2, 4)
    # plotshade(RMSE[4,:], sems[4,:], 0:dt:endI*dt, opts)
    # xlabel("time")
    # ylabel("root mean square error")
    # title("Variance RMSD")
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
