# Benchmark (Performance) a Solver

using BenchmarkTools;

include("GenerateQuadraticProgram.jl");

# const BENCHMARK_NUM_SAMPLES = 10;
# const BENCHMARK_NUM_EVALS   = 1;
# const BENCHMARK_RUN_TIME    = 90;

function GenerateElementsVector(numElementsMin = 20, numElementsMax = 2000; logSpace = true)
    if (logSpace)
        # Log Scale: More data points in the high values
        vNumElements = round.(Int, 10 .^ (LinRange(log10(numElementsMin), log10(numElementsMax), numDims)));
    else
        vNumElements = round.(Int, LinRange(numElementsMin, numElementsMax, numDims));
    end

    return vNumElements;
end

function BenchmarkSolver(solverFun, problemClass::ProblemClass, vNumElements, vNumConstraints; benchMarkSamples = 10, benchMarkEvals = 1, benchMarkSeconds = 120)
    
    # mR = Matrix{Float64}(undef, numDims, 5); #<! Min Time [Nano Sec], Median Time [Nano Sec], Max Time [Nano Sec], Number of Allocations, Allocations Size
    cBenchMark = Vector{Any}(undef, numDims);
    
    for ii in 1:numDims
        mP, vQ, mA, vL, vU = GenerateRandomQP(problemClass, vNumElements[ii], numConstraints = vNumConstraints[ii]);
        # Some problems might change the actual data
        numElements     = size(mP, 1);
        numConstraints  = size(mA, 1);

        vXX = zeros(numElements);

        sBenchMark = @benchmarkable $solverFun(vX, $mP, $vQ, $mA, $vL, $vU) setup = (vX = copy($vXX)); #<! We must interpolate the function as it is not in the global scope
        cBenchMark[ii] = run(sBenchMark, samples = benchMarkSamples, evals = benchMarkEvals, seconds = benchMarkSeconds);
    end
    
    return cBenchMark;
    
    
end

