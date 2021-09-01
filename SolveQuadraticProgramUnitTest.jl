# Unit Test for `SolveQuadraticProgram()`

using LinearAlgebra;
using SparseArrays;
# using Convex
# using SCS
# using ECOS #<! Seems to be very inaccurate for this problem
using Gurobi; #<! Solves problems which SCS struggles with
using OSQP;
using Plots;
using Random;
using StableRNGs;
using MAT;

seedNumber = 1234;
Random.seed!(seedNumber);
# Random.seed!(StableRNG(seedNumber), seedNumber);

include("GenerateQuadraticProgram.jl");
include("SolveQuadraticProgram.jl");
include("LinearSystemSolvers.jl");
include("SolveQuadraticProgramJump.jl");

# @enum ProblemClass randomQp = 1 inequalityConstrainedQp equalityConstrainedQp optimalControl portfolioOptimization lassoOptimization huberFitting supportVectorMachine isotonicRegression
@enum DataSource dataSourceGenerated = 1 dataSourceLoaded

## Parameters

# Simulaion
numSimulations  = 10;
numElements     = 500;
numConstraints  = 0; #<! Set to 0 for OSQP Paper dimensions

dataSource      = dataSourceGenerated;
dataFileName    = "QpModel.mat";

# problemClass    = rand(instances(ProblemClass));
problemClass    = supportVectorMachine;

# Solver
numIterations   = 5000;
ρ               = 0.1;
adptΡ           = true;
linSolverMode   = modeDirect;

if (dataSource == dataSourceGenerated)
    mP, vQ, mA, vL, vU = GenerateRandomQP(problemClass, numElements; numConstraints = numConstraints);
else
    dMatFile = matread(dataFileName); 
    mP = dMatFile["mP"];
    vQ = dropdims(dMatFile["vQ"]; dims = 2);
    mA = dMatFile["mA"];
    vL = dropdims(dMatFile["vL"]; dims = 2);
    vU = dropdims(dMatFile["vU"]; dims = 2);
end

# Some problems might change the actual data
numElements     = size(mP, 1);
numConstraints  = size(mA, 1);

vX = zeros(numElements);

fObjFun(vX) = 0.5 * dot(vX, mP, vX) + dot(vQ, vX);

hLinSolInit = LinMapsCgInit;
hLinSol     = LinMapsCg!;

osqPModel = OSQP.Model();
OSQP.setup!(osqPModel; P = mP, q = vQ, A = mA, l = vL, u = vU, rho = ρ, eps_abs = 1e-6, eps_rel = 1e-6, scaling = 0, );

runTimeJump = @elapsed vT = SolveQpJump(mP, vQ, mA, vL, vU; hOptFun = Gurobi.Optimizer);
runTimeOsqp = @elapsed osqpRes = OSQP.solve!(osqPModel);
runTime     = @elapsed convFlag = SolveQuadraticProgram!(vX, mP, vQ, mA, vL, vU, hLinSolInit, hLinSol; numIterations = numIterations, ρ = ρ, adptΡ = adptΡ);

maxAbsDev = norm(vT - vX, Inf);

display(scatter([vX, vT], title = "Solver Solution\nMax Absolute Deviation: $(maxAbsDev)\nRun Time: $(runTime) [Sec], Solver Mode: $(linSolverMode)", label = ["Solver" "Reference"]));
display(scatter([mA * vT, vL, vU], title = "Constraints Map", label = ["Solver" "Lower Boundary" "Upper Boundary"]));

println("The max absolute error is: $(maxAbsDev)");
println("The run time is: $(runTime) [Sec]");
println("The output flag is: $(convFlag)");
println("The JuMP.jl run time is: $(runTimeJump) [Sec]");
println("The OSQP run time is: $(runTimeOsqp) [Sec]");
println("The OSQP max absolute error is: $(norm(vT - osqpRes.x, Inf))");