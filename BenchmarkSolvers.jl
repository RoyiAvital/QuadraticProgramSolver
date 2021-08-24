# Unit Test for `SolveQuadraticProgram()`

using Random
using StableRNGs
using Plots


seedNumber = 1234;
Random.seed!(seedNumber);
# Random.seed!(StableRNG(seedNumber), seedNumber);

include("BenchmarkSolver.jl");
include("SolveQuadraticProgram.jl");

NS_TO_SEC_FCTR = 1e-9;

problemClass        = randomQp;
numElementsMin      = 750;
numElementsMax      = 750;
numConstraintsMin   = 50;
numConstraintsMax   = 50;
numDims             = 1;
logSpace            = false;

solverFun(vX, mP, vQ, mA, vL, vU) = SolveQuadraticProgram!(vX, mP, vQ, mA, vL, vU;
numIterations = 5000, ϵAbs = 1e-6, ϵRel = 1e-6,
ρ = 1e6, σ = 1e-6, α = 1.6, δ = 1e-6, adptΡ = true, 
fctrΡ = 5, numItrPolish = 0, linSolverMode = modeItertaive,
ϵPcg = 1e-7, numItrPcg = 15000, ϵMinres = 1e-6, numItrMinres = 500, numItrConv = 25);

currTime = time();
mR, vNumElements, vNumConstraints = BenchmarkSolver(solverFun, randomQp; numElementsMin = numElementsMin, numElementsMax = numElementsMax, numConstraintsMin = numConstraintsMin, numConstraintsMax = numConstraintsMax, numDims = numDims, logSpace = true);
runTime = time() - currTime;

println(runTime);

display(scatter(vNumElements, mR[:, 1] * NS_TO_SEC_FCTR, label = "Iterative Solver"));

solverFun(vX, mP, vQ, mA, vL, vU) = SolveQuadraticProgram!(vX, mP, vQ, mA, vL, vU;
numIterations = 5000, ϵAbs = 1e-6, ϵRel = 1e-6,
ρ = 1e6, σ = 1e-6, α = 1.6, δ = 1e-6, adptΡ = true, 
fctrΡ = 5, numItrPolish = 0, linSolverMode = modeDirect,
ϵPcg = 1e-7, numItrPcg = 15000, ϵMinres = 1e-6, numItrMinres = 500, numItrConv = 25);

currTime = time();
mR, vNumElements, vNumConstraints = BenchmarkSolver(solverFun, randomQp; numElementsMin = numElementsMin, numElementsMax = numElementsMax, numConstraintsMin = numConstraintsMin, numConstraintsMax = numConstraintsMax, numDims = numDims, logSpace = true);
runTime = time() - currTime;

println(runTime);

display(scatter(vNumElements, mR[:, 1] * NS_TO_SEC_FCTR, label = "Direct Solver"));

