using LinearAlgebra
using StatsFuns
using Distributions

function getKalman(A::Array{Float64,2}, C::Array{Float64,2}, totalI::Int, initVar::Array{Float64,2})
    if size(initVar,1)!=size(initVar,2)
        initVar = diagm(initVar)
    end
    Vars = Array{Array{Float64,2}, 1}(undef, totalI)
    Ks = Array{Array{Float64,2}, 1}(undef, totalI)
    Vars[1] = initVar
    #Get filter
    for i = 2:totalI
        Ks[i] = (A*Vars[i-1]*A'*C')/(C*A*Vars[i-1]*A'*C' .+ C*initVar*C')
        Vars[i] = (I - Ks[i]*C)*A*Vars[i-1]*A'
    end
    return Ks, Vars
end

struct KalmanModel{A2<:Array{Float64,2}, F<:Float64, I<:Int64, A1<:Array{Float64,1}, AA<:Array{Array{Float64,2},1}, FN<:Function}
    A::A2
    C::A2
    muPrior::A1
    initVar::A2
    a::F
    totalI::I
    Ks::AA
    Vars::AA
    runSim::FN
end


function runSim(Zs::Array{Array{Float64,1},1}, Ys::Array{Array{Float64,1},1}, KM::KalmanModel)
    Mus = Array{Float64,2}(undef, length(Zs[1])+1, KM.totalI)
    Mus[:,1] = vcat(KM.muPrior, 0)
    for i = 2:KM.totalI
        sum!(Mus[1:end-1,i], KM.A*Mus[1:end-1,i-1] + KM.Ks[i]*(Ys[i] - KM.C*KM.A*Mus[1:end-1,i-1]))
        Mus[end,i] = logaddexp(Mus[end,i-1], KM.getLLike(KM.C*Mus[1:end-1,i], KM.C*KM.Vars[i]*KM.C', Ys[i]))
    end
    Mus[1:end-1,:] -= hcat(Zs...)
    return Mus
end

function runSim(Mus::Array{Float64,2}, totalI::Int, A::Array{Float64,2}, C::Array{Float64,2},
                Ks::Array{Array{Float64,2},1}, Vars::Array{Array{Float64,2},1}, LLike::Function,
                Zs::Array{Array{Float64,1},1}, Ys::Array{Array{Float64,1},1})
    for i = 2:totalI
        sum!(Mus[1:end-1,i], A*Mus[1:end-1,i-1] + Ks[i]*(Ys[i] - C*A*Mus[1:end-1,i-1]))
        # Mus[end,i] = LLikeSum(Mus[end,i-1], C*Mus[1:end-1,i], C*Vars[i]*C', Ys[i])
        Mus[end,i] = LLike(C*Mus[1:end-1,i], C*Vars[i]*C', Ys[i])
    end
    Mus[1:end-1,:] -= hcat(Zs...)
    # cumsum!(Mus[end,:], Mus[end,:])
    accumulate!(logaddexp, Mus[end,:], Mus[end,:])
    return Mus
end

function getRunSim(A, C, muPrior, initVar, a, totalI, Ks, Vars)
    Mus = Array{Float64,2}(undef, length(muPrior)+1, totalI)
    Mus[:,1] = vcat(muPrior, 0)
    nCon = ((2*pi)^(-length(muPrior)/2))
    # LLikeSum = (last, Mu, Cov, Y)->logaddexp(last, log(nCon*(det(Cov)^(-1/2))*exp(-((Y-Mu)'*inv(Cov)*(Y-Mu))/2)))
    LLike = (Mu, Cov, Y)->log(nCon*(det(Cov)^(-1/2))*exp(-((Y-Mu)'*inv(Cov)*(Y-Mu))/2))
    return (Zs::Array{Array{Float64,1},1}, Ys::Array{Array{Float64,1},1}) -> runSim(Mus, totalI, A, C, Ks, Vars, LLike, Zs, Ys)
end


KalmanModel(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,1}, initVar::Array{Float64,1}, a::Float64, totalI::Int) =
    KalmanModel(A, C, muPrior, diagm(initVar), a, totalI, getKalman(A, C, totalI, diagm(initVar))...)
KalmanModel(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,1}, initVar::Array{Float64,2}, a::Float64, totalI::Int, Ks::Array{Array{Float64,2},1}, Vars::Array{Array{Float64,2},1}) =
    KalmanModel(A, C, muPrior, initVar, a, totalI, Ks, Vars, getRunSim(A, C, muPrior, initVar, a, totalI, Ks, Vars))
