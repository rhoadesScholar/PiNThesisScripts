# using Plots
using PyPlot
using Distributions
using Random
using LinearAlgebra
using Statistics


function runSim(muInit, A, C, oldMu, oldVar, sigmaEye, sigmaVest, a, inNoise, endI, sigmaExpected)
    err = zeros(size(muInit,1),endI)*NaN
    oldZ = muInit + [sqrt(a*sigmaEye)*[randn() randn()]; sqrt(a/(100*eps()))*[randn() randn()]; sqrt(a*sigmaVest)*[randn() randn()]]
    for i = 1:endI
        #Get actual latent
        oldZ = A*oldZ
        #Get observations

        Y = C*oldZ + rand!(inNoise,zeros(size(C*oldZ)))

        #Get filter()
        K = A*oldVar*A'*C'/(C*A*oldVar*A'*C' + sigmaExpected)
        oldVar = (I - K*C)*A*oldVar*A'
        oldMu = A*oldMu + K*(Y-C*A*oldMu);
        err[:,i] = sum((oldMu - oldZ).^2,dims=2)
    end
    return err
end

function plotshade(y, err, x, alpha, color, width)
    f = fill_between(x, y+err, y-err,color=color, alpha=alpha)

    # f.Annotation.LegendInformation.IconDisplayStyle = "off"
    plot(x,y,color=color,linewidth = width) ## change color | linewidth to adjust mean line()
end

function plotRMSE(RMSE, sems, dt, color)
    endI = size(RMSE,2)-1

    subplot(2, 2, 1)
    plotshade(RMSE[1,:], sems[1,:], 0:dt:endI*dt, .3, color, 2)
    xlabel("time")
    ylabel("root mean square error")
    title("Position RMSE")

    subplot(2, 2, 3)
    plotshade(RMSE[2,:], sems[2,:], 0:dt:endI*dt, .3, color, 2)
    xlabel("time")
    ylabel("root mean square error")
    title("Object Distance RMSE")

    subplot(2, 2, 4)
    plotshade(RMSE[3,:], sems[3,:], 0:dt:endI*dt, .3, color, 2)
    xlabel("time")
    ylabel("root mean square error")
    title("Velocity RMSE")
    return
end

function runBatchSim(argStr)
    # pyplot()

    #Batch Settings
    N = 1000

    #Sim Priors [position, object distance, velocity]
    P = [0 0]
    D = [10 10]
    V = [1 0]

    #Sim Settings
    a = 1
    endT = 100
    dt = .5
    sigmaVest = 1
    sigmaEye = 1

    #Plot Serrings
    color = [0 0 1]

    try
        eval(argStr)
    catch
        println("No argStr supplied.")
    end

    sigmaExpected = [sigmaEye 0; 0 sigmaVest]/dt
    endI = Integer(ceil(endT/dt))

    A = [1 0 dt; 0 1 -dt; 0 0 1]
    C = [0 1 0; 0 0 1]
    muInit = [P; D; V;]

    inNoise = MvNormal([0, 0], sigmaExpected)

    errs = zeros(size(muInit,1),endI,N)*NaN
    for n = 1:N#SHOULD BE PARFOR [if could start pool correctly...]
        oldMu = muInit
        oldVar = [a*sigmaEye 0 0; 0 a/(100*eps()) 0; 0 0 a*sigmaVest]
        errs[:,:,n] = runSim(muInit, A, C, oldMu, oldVar, sigmaEye, sigmaVest, a, inNoise, endI, sigmaExpected)
    end
    RMSE = sqrt.(dropdims(mean(errs, dims=3), dims=3))
    sems = dropdims(std(errs;mean=RMSE,dims=3)/sqrt(N), dims=3)

    plotRMSE(RMSE, sems, dt, color)

    return errs
end
