# Unit Test for `SolveQuadraticProgram()`

using LinearAlgebra
using SparseArrays
using Plots
using Random
using StableRNGs
using MAT
using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.samples = 5;
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10;

seedNumber = 1234;
Random.seed!(seedNumber);
# Random.seed!(StableRNG(seedNumber), seedNumber);

include("GenerateQuadraticProgram.jl");
include("SolveQuadraticProgram.jl");

# @enum ProblemClass randomQp = 1 inequalityConstrainedQp equalityConstrainedQp optimalControl portfolioOptimization lassoOptimization huberFitting supportVectorMachine isotonicRegression
@enum DataSource dataSourceGenerated = 1 dataSourceLoaded

## Parameters

# Simulaion
numElements     = 1000;
numConstraints  = 100;

dataSource      = dataSourceGenerated;
dataFileName    = "QpModel.mat";

# problemClass    = rand(instances(ProblemClass));
problemClass    = isotonicRegression;

# Solver
numIterations   = 5000;
ρ               = 1e6
adptΡ           = true;
linSolverMode   = modeItertaive;

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

sBenchMark = @benchmark convFlag = SolveQuadraticProgram!($vX, $mP, $vQ, $mA, $vL, $vU; numIterations = $numIterations, ρ = $ρ, adptΡ = $adptΡ, linSolverMode = $linSolverMode); setup = (vX .= $vXX);
display(sBenchMark);
println(convFlag);