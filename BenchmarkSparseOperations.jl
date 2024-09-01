# Compares Performance of Sparse Matrices with 32 Bit Indices vs 64 Bit Indices

using Random;
using StableRNGs;
using BenchmarkTools;
using Statistics;
using Plots;


seedNumber = 1234;
Random.default_rng() = StableRNG(seedNumber);

include("GenerateSpraseSymMatrix.jl");

NS_TO_SEC_FCTR          = 1e-9;
BYTE_TO_MEGA_BYTE_FCTR  = 2 ^ -20;

benchMarkSamples    = 10;
benchMarkEvals      = 1;
benchMarkSeconds    = 120;

vNumRows = [10; 30; 50; 70] .^ 2;


for (ii, matrixType) in enumerate(instances(MatrixType))
    currTime    = time();
    mA          = GenerateSpraseSymMatrix(matrixtype; numRows = vNumRows[ii]);


    sBenchMark      = @benchmarkable $solverFun(vX, $mP, $vQ, $mA, $vL, $vU) setup = (vX = copy($vXX)); #<! We must interpolate the function as it is not in the global scope
    cBenchMark[ii]  = run(sBenchMark, samples = benchMarkSamples, evals = benchMarkEvals, seconds = benchMarkSeconds);
    runTime     = time() - currTime;
end


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
