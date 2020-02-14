struct SimWorld{A2<:Array{Float64,2}, F<:Float64, I<:Int64, A1<:Array{Float64,1}, AA<:Array{Array{Float64,2},1}, FN<:Function}
    A::A2
    C::A2
    muInit::A1
    endT::F
    dt::F
    endI::I
    allT::A1
    allAs::AA
    Zs::FN
end
SimWorld(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,1}, endT::Float64=1000., dt::Float64=.1) =
    SimWorld(A, C, muPrior, endT, dt, Integer(ceil(endT/dt)))
SimWorld(A::Array{Float64,2}, C::Array{Float64,2}, muPrior::Array{Float64,1}, endT::Float64, dt::Float64, endI::Int) =
    SimWorld(A, C, muPrior, endT, dt, endI,
                Array{Float64,1}(0:dt:endI*dt),
                [A^t for t in 0:endI], (A::Array{Float64,2},Z::Array{Float64,1})->A*Z)



    Ys::FN
    Z::FN2
end
FullWorld(static::StaticWorld, simOpts::SimOpts) =
    FullWorld(static::StaticWorld, simOpts.a,
    diagm(simOpts.a*simOpts.sigmas), (static.C*diagm(simOpts.sigmas)*static.C')/static.dt)
FullWorld(static::StaticWorld, a::Float64, initVar::Array{Float64,2}, sigmaIn::Array{Float64,2}) =
    FullWorld(static, a, sigmaIn,
    (Z::Array{Float64,1})->rand!(MvNormal(zeros(size(sigmaIn,1)), sigmaIn), similar(static.C*static.muPrior)) + static.C*Z,
    getKalman(A::Array{Float64,2}, C::Array{Float64,2}, totalI::Int, initVar::Array{Float64,2})...,
    ()->(static.muPrior + sqrt.(diag(initVar)).*randn(size(static.muPrior))))

Z = collect(Iterators.repeated(kworld.Z(),kworld.static.endI+1));
    Zs = kworld.static.Zs.(kworld.static.allAs, Z);
    Ys = kworld.Ys.(Zs);#[kworld.static.C*z .+ rand!(kworld.noise, similar(kworld.static.C*Z[1])) for z in Zs]
