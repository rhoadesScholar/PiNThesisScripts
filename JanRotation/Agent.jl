include("KalmanModel.jl")
include("SimWorld.jl")

struct Agent{KMs<:Array{KalmanModel,1}, SWs<:Array{SimWorld,1}}
    models::KMs
    worlds::SWs
end
