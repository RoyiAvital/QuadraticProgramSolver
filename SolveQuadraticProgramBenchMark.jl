# Unit Test for `SolveQuadraticProgram()`

using LinearAlgebra;
using SparseArrays;
using Plots;
using Random;
using StableRNGs;
using MAT;
using BenchmarkTools;

seedNumber = 1234;
Random.seed!(seedNumber);
# Random.seed!(StableRNG(seedNumber), seedNumber);

include("GenerateQuadraticProgram.jl");
include("SolveQuadraticProgram.jl");

# @enum ProblemClass randomQp = 1 inequalityConstrainedQp equalityConstrainedQp optimalControl portfolioOptimization lassoOptimization huberFitting supportVectorMachine isotonicRegression
@enum DataSource dataSourceGenerated = 1 dataSourceLoaded

## Parameters

# Simulaion
numElements     = 800;
numConstraints  = 400;

dataSource      = dataSourceGenerated;
dataFileName    = "QpModel.mat";

# problemClass    = rand(instances(ProblemClass));
problemClass    = randomQp;

# Solver
numIterations   = 7500;
ρ               = 1e6
adptΡ           = true;
linSolverMode   = modeItertaive;
numItrPolish    = 0;

if (dataSource == dataSourceGenerated)
    mP, vQ, mA, vL, vU = GenerateRandomQP(problemClass, numElements, numConstraints = numConstraints);
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

vXX = zeros(numElements);

sBenchMark = @benchmarkable SolveQuadraticProgram!(vX, $mP, $vQ, $mA, $vL, $vU; numIterations = $numIterations, ρ = $ρ, adptΡ = $adptΡ, linSolverMode = $linSolverMode, numItrPolish = $numItrPolish) setup = (vX = copy($vXX));
currTime    = time();
tuRunResult = BenchmarkTools.run_result(sBenchMark, samples = 10, evals = 1, seconds = 150); #<! A trick to get result as well
runTime     = time() - currTime;

display(tuRunResult[1]);
println("\nTotal Run Time: $runTime [Sec]");
println("The Alogrihtm Convergence Status: $(tuRunResult[2])\n\n");