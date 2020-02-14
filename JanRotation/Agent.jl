struct Agent(KMs<:Array{KalmanModel,1}, SMs<:Array{SimWorld,1})
    models::KMs
    worlds::SMs
    
end
