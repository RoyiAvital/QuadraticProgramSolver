# Benchmark 

using Plots;
using Random;
using Statistics;
using BenchmarkTools;

Random.seed!(1234);

include("GenerateQuadraticProgram.jl");
include("LinearSystemSolvers.jl");

NS_TO_SEC_FCTR          = 1e-9;
BYTE_TO_MEGA_BYTE_FCTR  = 2 ^ -20;

## Parameters

# Simulaion
vNumElements    = [250; 500];
vNumConstraints = [125; 250];

# problemClass    = rand(instances(ProblemClass));
problemClass    = randomQp;

## Generating Model
ρ = 1e6;
σ = 1e-6;

# Solver Parameters
numFactor       = 3;
numIterations   = 50;
ϵSolver         = 1e-6;
numItrSolver    = 1e-6;

# Benchmark
benchMarkSamples    = 100;
benchMarkEvals      = 1;
benchMarkSeconds    = 2;

# Solution Threshold
ϵSol = 1e-5;





numDims = size(vNumElements, 1);






vF      = Vector{Any}(undef, 0);
vLabel  = Vector{Any}(undef, 0);

# LaLdlt!(vX, mP, mA, vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 1e-6, ρ = 1e6, σ = 1e-6);
# ItrSolCg!(vX, mP, mA, vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500, ρ = 1e6, σ = 1e-6);
# KrylovCg!(vX, mP, mA, vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500, ρ = 1e6, σ = 1e-6);
# KrylovCr!(vX, mP, mA, vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500, ρ = 1e6, σ = 1e-6);
# KrylovCgLanczos!(vX, mP, mA, vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500, ρ = 1e6, σ = 1e-6);
# LinOpCg!(vX, mP, mA, vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500, ρ = 1e6, σ = 1e-6);
# KrylovTriCg!(vX, mP, mA, vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500, ρ = 1e6, σ = 1e-6);
# KrylovTriMr!(vX, mP, mA, vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500, ρ = 1e6, σ = 1e-6);


push!(vF, (vX, mP, mA, vV) -> LaLdlt!(vX, mP, mA, vV, ρ, σ; numFactor = numFactor, numIterations = numIterations, ϵSolver = ϵSolver, numItrSolver = numItrSolver));
push!(vLabel, "LDL Factorization");
push!(vF, (vX, mP, mA, vV) -> ItrSolCg!(vX, mP, mA, vV, ρ, σ; numFactor = numFactor, numIterations = numIterations, ϵSolver = ϵSolver, numItrSolver = numItrSolver));
push!(vLabel, "IterativeSolvers CG");
push!(vF, (vX, mP, mA, vV) -> KrylovCg!(vX, mP, mA, vV, ρ, σ; numFactor = numFactor, numIterations = numIterations, ϵSolver = ϵSolver, numItrSolver = numItrSolver));
push!(vLabel, "Krylov CG");
push!(vF, (vX, mP, mA, vV) -> KrylovCr!(vX, mP, mA, vV, ρ, σ; numFactor = numFactor, numIterations = numIterations, ϵSolver = ϵSolver, numItrSolver = numItrSolver));
push!(vLabel, "Krylov CR");
push!(vF, (vX, mP, mA, vV) -> KrylovCgLanczos!(vX, mP, mA, vV, ρ, σ; numFactor = numFactor, numIterations = numIterations, ϵSolver = ϵSolver, numItrSolver = numItrSolver));
push!(vLabel, "Kryloc Lanczos");
push!(vF, (vX, mP, mA, vV) -> LinOpCg!(vX, mP, mA, vV, ρ, σ; numFactor = numFactor, numIterations = numIterations, ϵSolver = ϵSolver, numItrSolver = numItrSolver));
push!(vLabel, "LinearOperators CG");
push!(vF, (vX, mP, mA, vV) -> KrylovTriCg!(vX, mP, mA, vV, ρ, σ; numFactor = numFactor, numIterations = numIterations, ϵSolver = ϵSolver, numItrSolver = numItrSolver));
push!(vLabel, "Kryloc TriCG");
push!(vF, (vX, mP, mA, vV) -> KrylovTriMr!(vX, mP, mA, vV, ρ, σ; numFactor = numFactor, numIterations = numIterations, ϵSolver = ϵSolver, numItrSolver = numItrSolver));
push!(vLabel, "Krylov TriMr");

numSolvers  = size(vF, 1);
cBenchMark  = Vector{Any}(undef, numSolvers);
vCorrSol    = falses(numSolvers);

tR = zeros(numDims, numSolvers, 5); #<! Min Time [Nano Sec], Median Time [Nano Sec], Max Time [Nano Sec], Number of Allocations, Allocations Size
for ii = 1:numDims
    mP, vQ, mA, vL, vU = GenerateRandomQP(problemClass, vNumElements[ii], numConstraints = vNumConstraints[ii]);
    
    # Some problems might change the actual data
    numElements     = size(mP, 1);
    numConstraints  = size(mA, 1);
    
    vX = randn(numElements);
    vV = randn(numElements + numConstraints);
    
    currTime    = time();
    vXRef       = (mP + (σ * sparse(I, numElements, numElements)) + (ρ * (mA' * mA))) \ vX;
    runTime     = time() - currTime;
    
    println("\nThe Reference Run Time: $runTime [Sec]\n");
    for jj = 1:numSolvers
        sBenchMark      = @benchmarkable vF[ii](vXX, $mP, $vQ, $mA, vV) setup = (vXX = copy($vX)); #<! We must interpolate the function as it is not in the global scope
        cBenchMark[ii]  = run(sBenchMark, samples = benchMarkSamples, evals = benchMarkEvals, seconds = benchMarkSeconds);
        vCorrSol[ii]    = norm(vXX - vXRef, 2) <= (numElements * ϵThr);
        
        mR[ii, 1] = min(cBenchMark[ii].times...);
        mR[ii, 2] = max(cBenchMark[ii].times...);
        mR[ii, 3] = median(cBenchMark[ii].times);
        mR[ii, 4] = cBenchMark[ii].allocs;
        mR[ii, 5] = cBenchMark[ii].memory;
    end
end

dataTitle = ["Solvers Min Run Time" "Solvers Max Run Time" "Solvers Median Run Time" "Solvers Allocations" "Solvers Allocations" "Solvers Success"]

for ii = 1:3
    display(scatter(1:numSolvers, mR[:, ii] * NS_TO_SEC_FCTR, title = dataTitle[ii], label = solversLabels, xlabel = "Solver", ylabel = "Time [Sec]"));
end

ii = 4;
display(scatter(1:numSolvers, mR[:, ii], title = dataTitle[ii], label = solversLabels, xlabel = "Solver", ylabel = "Number of Allocations"));

ii = 5;
display(scatter(1:numSolvers, mR[:, ii] * NS_TO_SEC_FCTR, title = dataTitle[ii], label = solversLabels, xlabel = "Solver", ylabel = "Size [MB]"));

ii = 6;
display(scatter(1:numSolvers, vCorrSol, title = dataTitle[ii], label = solversLabels, xlabel = "Solver", ylabel = "Solver Success Flag"));

# @benchmark LaLdlt!($vX, $mP, $mA, $vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 1e-6, ρ = 1e6, σ = 1e-6);
# @benchmark ItrSolCg!($vX, $mP, $mA, $vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500, ρ = 1e6, σ = 1e-6);
# @benchmark KrylovCg!($vX, $mP, $mA, $vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500, ρ = 1e6, σ = 1e-6);
# @benchmark KrylovCr!($vX, $mP, $mA, $vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500, ρ = 1e6, σ = 1e-6);
# @benchmark KrylovCgLanczos!($vX, $mP, $mA, $vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500, ρ = 1e6, σ = 1e-6);
# @benchmark LinOpCg!($vX, $mP, $mA, $vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500, ρ = 1e6, σ = 1e-6);
# @benchmark KrylovTriCg!($vX, $mP, $mA, $vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500, ρ = 1e6, σ = 1e-6);
# @benchmark KrylovTriMr!($vX, $mP, $mA, $vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500, ρ = 1e6, σ = 1e-6);

# ρ = 1e6;
# σ = 1e-6;

# vT = zeros(numConstraints); #<! Buffer
# opL = LinearOperator(Float64, numElements, numElements, true, true, (vU, vW, α, β) -> begin
#     mul!(vT, mA, vW);
#     mul!(vU, transpose(mA), vT);
#     mul!(vU, mP, vW, one(Float64), ρ);
#     vU .= vU .+ (σ .* vW);
# end);

# mL = mP + (σ * sparse(I, numElements, numElements)) + (ρ * (transpose(mA) * mA));

# vX = rand(numElements);

# println(norm((opL * vX) - (mL * vX)))

# vT = mA * vX;
# vU = transpose(mA) * vT;
# vU .= mP * vX .+ ρ .* vU;
# vU .= vU .+ (σ * vX);

# println(norm(vU - mL * vX))









