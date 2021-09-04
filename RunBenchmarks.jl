# Unit Test for `SolveQuadraticProgram()`

using LinearAlgebra;
using SparseArrays;
using Random;
using StableRNGs;
using Dates;
using BenchmarkTools;

seedNumber = 1234;
Random.seed!(seedNumber);
# Random.seed!(StableRNG(seedNumber), seedNumber);
Random.default_rng() = StableRNG(seedNumber);

include("GenerateQuadraticProgram.jl");
include("SolveQuadraticProgram.jl");
include("LinearSystemSolvers.jl");

# @enum ProblemClass randomQp = 1 inequalityConstrainedQp equalityConstrainedQp optimalControl portfolioOptimization lassoOptimization huberFitting supportVectorMachine isotonicRegression

## Parameters

# Simulaion
numSimulations  = 5;
mNumElements = [    0010 0100;
0010 0100;
0010 0100;
0010 0100;
0010 0100;
0010 0100;
0010 0100;
0010 0100;
0010 0100;];

mNumConstraints = [ 0000 0000;
0000 0000;
0000 0000;
0000 0000;
0000 0000;
0000 0000;
0000 0000;
0000 0000;
0000 0000;]; #<! Set to 0 for OSQP Paper dimensions

# Solver
numIterations   = 50000; #<! Enough for Primal Dual / ADMM Convergence
numItrPolish    = 0;
hLinSolInit     = FacLdlInit;
hLinSol!        = FacLdl!;

hSolverFun(vX, mP, vQ, mA, vL, vU) = SolveQuadraticProgram!(vX, mP, vQ, mA, vL, vU, hLinSolInit, hLinSol!; numIterations = numIterations, numItrPolish = numItrPolish);

solverLabel = "LDLFactorizations.jl Solver";
codeVersion = "0.1.000";
cpuInfo     = rstrip(lstrip(Sys.cpu_info()[1].model));
benchDate   = Dates.format(now(Dates.UTC), "yyyy_mm_dd_SS_MM_HH");

# Benchmarking
benchMarkSamples    = 15;
benchMarkEvals      = 1;
benchMarkSeconds    = 120;

numTests    = numSimulations * length(mNumElements);
vTestData   = Vector{Any}(undef, 4 * numTests + 4);


vTestData[1] = solverLabel;
vTestData[2] = codeVersion;
vTestData[3] = cpuInfo;
vTestData[4] = benchDate;

testIdx = 5;

for problemClass ∈ instances(ProblemClass)
    for iDim ∈ 1:size(mNumElements, 2)
        for iSim ∈ 1:numSimulations
            mP, vQ, mA, vL, vU = GenerateRandomQP(problemClass, mNumElements[Int(problemClass), iDim]; numConstraints = mNumConstraints[Int(problemClass), iDim]);
            # Some problems might change the actual data
            numElements     = size(mP, 1);
            numConstraints  = size(mA, 1);
            
            vXX = zeros(numElements);

            sBenchMark              = @benchmarkable $hSolverFun(vX, $mP, $vQ, $mA, $vL, $vU) setup = (vX = copy($vXX)); #<! We must interpolate the function as it is not in the global scope
            cBenchMark, convFlag    = BenchmarkTools.run_result(sBenchMark, samples = benchMarkSamples, evals = benchMarkEvals, seconds = benchMarkSeconds);
            
            vTestData[testIdx + 0] = min(cBenchMark.times...);
            vTestData[testIdx + 1] = cBenchMark.allocs;
            vTestData[testIdx + 2] = cBenchMark.memory;
            vTestData[testIdx + 3] = convFlag ≠ convNumItr;

            global testIdx += 4;
            println(testIdx);
        end
    end
end