# Benchmark (Performance) a Solver

using BenchmarkTools

include("GenerateQuadraticProgram.jl");

BenchmarkTools.DEFAULT_PARAMETERS.samples = 5;
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 100;

function BenchmarkSolver(solverFun, problemClass::ProblemClass; numElementsMin = 20, numElementsMax = 2000, numConstraintsMin = 10, numConstraintsMax = 1000, numDims = 5, logSpace = true)
    
    if (logSpace)
        # Log Scale: More data points in the high values
        vNumElements    = round.(Int, 10 .^ (LinRange(log10(numElementsMin), log10(numElementsMax), numDims)));
        vNumConstraints =  round.(Int, 10 .^ (LinRange(log10(numConstraintsMin), log10(numConstraintsMax), numDims)));
    else
        vNumElements    = round.(Int, LinRange(numElementsMin, numElementsMax, numDims));
        vNumConstraints = round.(Int, LinRange(numConstraintsMin, numConstraintsMax, numDims));
    end
    
    mR = Matrix{Float64}(undef, numDims, 3); #<! Time [Nano Sec], Number of Allocations, Allocations Size
    
    for ii in 1:numDims
        mP, vQ, mA, vL, vU = GenerateRandomQP(problemClass, vNumElements[ii], numConstraints = vNumConstraints[ii]);
        # Some problems might change the actual data
        numElements     = size(mP, 1);
        numConstraints  = size(mA, 1);

        vXX = zeros(numElements);

        sBenchMark = @benchmark solverFun(vX, $mP, $vQ, $mA, $vL, $vU) setup = (vX = copy($vXX););
        mR[ii, 1] = min(sBenchMark.times...); #<! Minimum Running Time [Nano Sec]
        mR[ii, 2] = sBenchMark.allocs; #<! Number of allocations
        mR[ii, 3] = sBenchMark.memory; #<! Allocation Size [Bytes]
    end
    
    return mR, vNumElements, vNumConstraints;
    
    
end

