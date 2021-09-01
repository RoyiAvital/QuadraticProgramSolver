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
include("LinearSystemSolvers.jl");

NS_TO_SEC_FCTR          = 1e-9;
BYTE_TO_MEGA_BYTE_FCTR  = 2 ^ -20;

problemClass        = randomQp;
numElementsMin      = 200;
numElementsMax      = 1200;
numConstraintsMin   = 0;
numConstraintsMax   = 0;
numDims             = 5;
logSpace            = false;

solversLabels = [];

# hF(vX, mP, vQ, mA, vL, vU) = SolveQuadraticProgram!(vX, mP, vQ, mA, vL, vU, LaLdlInit, LaLdl!;
# numIterations = 5000, ϵAbs = 1e-6, ϵRel = 1e-6,
# ρ = 1, σ = 1e-6, α = 1.6, δ = 1e-6, adptΡ = true, 
# fctrΡ = 5, numItrConv = 25, numItrPolish = 0, ϵMinres = 1e-6, numItrMinres = 500);
# push!(solversLabels, "LA LDL");

# hG(vX, mP, vQ, mA, vL, vU) = SolveQuadraticProgram!(vX, mP, vQ, mA, vL, vU, QDLdlInit, QDLdl!;
# numIterations = 5000, ϵAbs = 1e-6, ϵRel = 1e-6,
# ρ = 1, σ = 1e-6, α = 1.6, δ = 1e-6, adptΡ = true, 
# fctrΡ = 5, numItrConv = 25, numItrPolish = 0, ϵMinres = 1e-6, numItrMinres = 500);
# push!(solversLabels, "QD LDL");

# hH(vX, mP, vQ, mA, vL, vU) = SolveQuadraticProgram!(vX, mP, vQ, mA, vL, vU, FacLdlInit, FacLdl!;
# numIterations = 5000, ϵAbs = 1e-6, ϵRel = 1e-6,
# ρ = 1, σ = 1e-6, α = 1.6, δ = 1e-6, adptΡ = true, 
# fctrΡ = 5, numItrConv = 25, numItrPolish = 0, ϵMinres = 1e-6, numItrMinres = 500);
# push!(solversLabels, "LDLFac LDL");

hF(vX, mP, vQ, mA, vL, vU) = SolveQuadraticProgram!(vX, mP, vQ, mA, vL, vU, LinOpCgInit, LinOpCg!;
numIterations = 5000, ϵAbs = 1e-6, ϵRel = 1e-6,
ρ = 1, σ = 1e-6, α = 1.6, δ = 1e-6, adptΡ = true, 
fctrΡ = 5, numItrConv = 25, numItrPolish = 0, ϵMinres = 1e-6, numItrMinres = 500);
push!(solversLabels, "LinearOperators.jl");

hG(vX, mP, vQ, mA, vL, vU) = SolveQuadraticProgram!(vX, mP, vQ, mA, vL, vU, LinMapsCgInit, LinMapsCg!;
numIterations = 5000, ϵAbs = 1e-6, ϵRel = 1e-6,
ρ = 1, σ = 1e-6, α = 1.6, δ = 1e-6, adptΡ = true, 
fctrΡ = 5, numItrConv = 25, numItrPolish = 0, ϵMinres = 1e-6, numItrMinres = 500);
push!(solversLabels, "LinearMaps.jl");

vFun = [hF; hG];

numSolvers      = size(vFun, 1);
solversLabels   = reshape(solversLabels, 1, numSolvers); #<! For plotting

vNumElements    = GenerateElementsVector(numElementsMin, numElementsMax; logSpace = logSpace);
vNumConstraints = GenerateElementsVector(numConstraintsMin, numConstraintsMax; logSpace = logSpace);

tR = zeros(numDims, numSolvers, 5); #<! Min Time [Nano Sec], Median Time [Nano Sec], Max Time [Nano Sec], Number of Allocations, Allocations Size


for ii = 1:numSolvers
    currTime    = time();
    cBenchMark  = BenchmarkSolver(vFun[ii], problemClass, vNumElements, vNumConstraints);
    runTime     = time() - currTime;

    println(cBenchMark);
    
    println("\nThe Run Time of the $ii -th Solver ($(solversLabels[ii])): $runTime [Sec]\n");
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
