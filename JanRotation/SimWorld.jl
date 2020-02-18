using LinearAlgebra
using Distributions
using Random

struct SimWorld{A2<:Array{Float64,2}, F<:Float64, I<:Int64, A1<:Array{Float64,1}, AA<:Array{Array{Float64,2},1}, FN<:Function}
    A::A2
    C::A2
    muInit::A1
    initVar::A1
    endT::F
    dt::F
    endI::I
    allT::A1
    allAs::AA
    getStates::FN
end

function getStates(SW::SimWorld)
    Z = (SW.muInit .+ sqrt.(SW.initVar).*randn(size(SW.muInit)))
    Zs = [A*Z for A in SW.allAs]
    Ys = [SW.C*z .+ rand!(MvNormal(zeros(size(SW.C*SW.C',1)), SW.C*diagm(SW.initVar)*SW.C'), similar(SW.C*SW.muInit)) for z in Zs];
    return Zs::Array{Array{Float64,1},1}, Ys::Array{Array{Float64,1},1}
end

SimWorld(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,1}, initVar::Array{Float64,1}, endT::Float64=1000., dt::Float64=.1) =
    SimWorld(A, C, muPrior, initVar, endT, dt, Integer(ceil(endT/dt)))
SimWorld(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,1}, initVar::Array{Float64,1}, endT::Float64, dt::Float64, endI::Int) =
    SimWorld(A, C, muPrior, initVar, endT, dt, endI,
                Array{Float64,1}(0:dt:endI*dt),
                [A^t for t in 0:endI],
                getStates)
