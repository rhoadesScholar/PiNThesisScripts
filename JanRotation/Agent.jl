include("KalmanModel.jl")
include("SimWorld.jl")

struct Agent{KMs<:Array{KalmanModel,1}, FN<:Function}#, SWs<:Array{SimWorld,1}}
    models::KMs
    choice::FN
    # worlds::SWs
end
