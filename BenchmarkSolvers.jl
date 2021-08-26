# Unit Test for `SolveQuadraticProgram()`

using Random;
using StableRNGs;
using Statistics;
using Plots;


seedNumber = 1234;
Random.seed!(seedNumber);
# Random.seed!(StableRNG(seedNumber), seedNumber);

include("BenchmarkSolver.jl");
include("SolveQuadraticProgram.jl");

NS_TO_SEC_FCTR          = 1e-9;
BYTE_TO_MEGA_BYTE_FCTR  = 2 ^ -20;

problemClass        = randomQp;
numElementsMin      = 250;
numElementsMax      = 750;
numConstraintsMin   = 125;
numConstraintsMax   = 375;
numDims             = 3;
logSpace            = false;

hF(vX, mP, vQ, mA, vL, vU) = SolveQuadraticProgram!(vX, mP, vQ, mA, vL, vU;
numIterations = 5000, ϵAbs = 1e-6, ϵRel = 1e-6,
ρ = 1e6, σ = 1e-6, α = 1.6, δ = 1e-6, adptΡ = true, 
fctrΡ = 5, numItrPolish = 0, linSolverMode = modeItertaive,
ϵPcg = 1e-7, numItrPcg = 5000, ϵMinres = 1e-6, numItrMinres = 500, numItrConv = 25);

hG(vX, mP, vQ, mA, vL, vU) = SolveQuadraticProgram!(vX, mP, vQ, mA, vL, vU;
numIterations = 5000, ϵAbs = 1e-6, ϵRel = 1e-6,
ρ = 1e6, σ = 1e-6, α = 1.6, δ = 1e-6, adptΡ = true, 
fctrΡ = 5, numItrPolish = 0, linSolverMode = modeDirect,
ϵPcg = 1e-7, numItrPcg = 5000, ϵMinres = 1e-6, numItrMinres = 500, numItrConv = 25);

vFun = [hF; hG];

solversLabels = ["Iterative Solver" "Direct Solver"];

numSolvers = size(vFun, 1);

vNumElements    = GenerateElementsVector(numElementsMin, numElementsMax; logSpace = logSpace);
vNumConstraints = GenerateElementsVector(numConstraintsMin, numConstraintsMax; logSpace = logSpace);

tR = zeros(numDims, numSolvers, 5); #<! Min Time [Nano Sec], Median Time [Nano Sec], Max Time [Nano Sec], Number of Allocations, Allocations Size


for ii = 1:numSolvers
    currTime    = time();
    cBenchMark  = BenchmarkSolver(vFun[ii], problemClass, vNumElements, vNumConstraints);
    runTime     = time() - currTime;

    println(cBenchMark);
    
    println("\nThe Run Time of the $ii -th Solver $runTime [Sec]\n");
    for jj = 1:numDims
        tR[jj, ii, 1] = min(cBenchMark[jj].times...);
        tR[jj, ii, 2] = max(cBenchMark[jj].times...);
        tR[jj, ii, 3] = median(cBenchMark[jj].times);
        tR[jj, ii, 4] = cBenchMark[jj].allocs;
        tR[jj, ii, 5] = cBenchMark[jj].memory;
    end
end

dataTitle = ["Solvers Min Run Time" "Solvers Max Run Time" "Solvers Median Run Time" "Solvers Allocations" "Solvers Allocations"]

for ii = 1:3
    display(scatter(vNumElements, tR[:, :, ii] * NS_TO_SEC_FCTR, title = dataTitle[ii], label = solversLabels, xlabel = "Dimension", ylabel = "Time [Sec]"));
end

ii = 4;
display(scatter(vNumElements, tR[:, :, ii], title = dataTitle[ii], label = solversLabels, xlabel = "Dimension", ylabel = "Number of Allocations"));

ii = 5;
display(scatter(vNumElements, tR[:, :, ii] * NS_TO_SEC_FCTR, title = dataTitle[ii], label = solversLabels, xlabel = "Dimension", ylabel = "Size [MB]"));
