
function runSim(Ys::Array{Array{Float64,2},1}, self::KalmanModel)
    Mus = Array{Float64,2}(undef, length(Ys[1])+1, self.totalI)
    Mus[1] = vcat(self.muPrior, 0)
    for i = 2:totalI
        Mus[i] = vcat(self.A*Mus[i-1] + self.Ks[i]*(Ys[i] - self.C*self.A*Mus[i-1]), logaddexp(Mus[i-1], stableLogPDF(self.C*Mus[i], self.C*self.Vars[i]*self.static.C', Ys[i])))
    end
    return hcat(Mus...)
end

function getKalman(A::Array{Float64,2}, C::Array{Float64,2}, totalI::Int, initVar::Array{Float64,2})
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

struct KalmanModel(A2<:Array{Float64,2}, F<:Float64, I<:Int64, A1<:Array{Float64,1}, AA<:Array{Array{Float64,2},1}, FN<:Function)
    A::A2
    C::A2
    muPrior::A1
    initVar::A2
    a::F
    Ks::AA
    Vars::AA
    MusLL::FN
end

KalmanModel(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,1}, initVar::Array{Float64,2}, a::Float64, totalI::Int) =
    KalmanModel(A, C, muPrior, initVar, a, totalI, getKalman(A, C, totalI, initVar)...)
KalmanModel(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,1}, initVar::Array{Float64,2}, a::Float64, totalI::Int,
        Ks::Array{Array{Float64,2},1}, Vars::Array{Array{Float64,2},1}) =
    KalmanModel(A, C, muPrior, initVar, a, totalI, Ks, Vars,
        (Ys::Array{Array{Float64,2},1})->runSim(Ys, self))
