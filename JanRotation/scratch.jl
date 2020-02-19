include("Agent.jl")
include("linspecer.jl")
using Distributed

dt = .1;
A1 = [1. 0 0 0 0 0 dt 0;
      0 1 0 0 0 0 0 dt;
      0 0 1 0 dt 0 -dt 0;
      0 0 0 1 0 dt 0 -dt;
      0 0 0 0 1 0 0 0;
      0 0 0 0 0 1 0 0;
      0 0 0 0 0 0 1 0;
      0 0 0 0 0 0 0 1];

 A2 = [1. 0 0 0 0 0 dt 0;
       0 1 0 0 0 0 0 dt;
       0 0 1 0 0 0 -dt 0;
       0 0 0 1 0 0 0 -dt;
       0 0 0 0 1 0 0 0;
       0 0 0 0 0 1 0 0;
       0 0 0 0 0 0 1 0;
       0 0 0 0 0 0 0 1];

As = Array{Array{Float64,2},1}(undef,2);
As[1] = A1;
As[2] = A2;

C = [0. 0 1 0 0 0 0 0;
     0. 0 0 1 0 0 0 0;
     0. 0 0 0 0 0 1 0;
     0. 0 0 0 0 0 0 1];
muPrior = [0., 0, 10., 0, 0., 1, 1., 0];
endT = 100.;
initVar = [1., 1, 100., 100, 1., 1, 1., 1];
emitVar = [1., 1, 100., 100, 1., 1, 1., 1];
a = 1.;

KMs = empty!(Array{KalmanModel,1}(undef,1));
for A in As
      push!(KMs, KalmanModel(A, C, muPrior, initVar, a, Integer(ceil(endT/dt))+1));
end

SWs = empty!(Array{SimWorld,1}(undef,1));
for A in As
      push!(SWs, SimWorld(A, C, muPrior, emitVar, endT, dt))
end

agent = Agent(KMs, SWs)

N=100
MusLL = Array{Array{Float64,2},3}(undef, length(SWs), length(agent.models), N)
@everywhere k=0;
for SW in SWs
      @everywhere k+=1
      @simd for i = 1:N
            j = 0;
            Zs, Ys = SW.getStates(SW)
            @simd for KM in agent.models
                  j+=1
                  MusLL[k,j,i] = KM.runSim(Zs, Ys)
            end
      end
end

findmax!(maxval, ind, A)



___________________________
static = StaticWorld(A, C, muPrior, endT, dt);

flexSigValues = [10, 1, .1]
flexSigInd = 4
flexVarName = "sigmaVest"

v=1

simNames = Array{String, 1}(undef,2)
simNames[1] = "Moving"
simNames[2] = "Still"

set_zero_subnormals(true)
variationNum = length(simNames)
colors = linspecer(variationNum)

simOpts = SimOpts(sigmas, 1000)
plotOpts = PlotOpts(simNames[v], colors[v,:])
kworld = FullWorld(static, simOpts);
