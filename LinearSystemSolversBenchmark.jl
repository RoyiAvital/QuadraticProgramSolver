# Unit Test for `SolveQuadraticProgram()`

using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Krylov
using Plots
using Random
using MAT
using BenchmarkTools
using LinearOperators

Random.seed!(1234);

include("GenerateQuadraticProgram.jl");
include("LinearSystemSolvers.jl");

## Parameters

# Simulaion
numElements     = 250;
numConstraints  = 20;

# problemClass    = rand(instances(ProblemClass));
problemClass    = randomQp;

## Generating Model
mP, vQ, mA, vL, vU = GenerateRandomQP(problemClass, numElements, numConstraints = numConstraints);
ρ = 1e6;
σ = 1e-6;

# Solver Parameters
numFactor       = 3;
numIterations   = 50;
ϵSolver         = 1e-6;
numItrSolver    = 1e-6;


# Some problems might change the actual data
numElements     = size(mP, 1);
numConstraints  = size(mA, 1);

vX = randn(numElements);
vV = randn(numElements + numConstraints);

currTime    = time();
vXRef       = (mP + (σ * sparse(I, numElementsX, numElementsX)) + (ρ * (mA' * mA))) \ vX;
runTime     = time() - currTime;

println("\The Reference Run Time: $runTime [Sec]");

# LaLdlt!(vX, mP, mA, vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 1e-6, ρ = 1e6, σ = 1e-6);
# ItrSolCg!(vX, mP, mA, vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500, ρ = 1e6, σ = 1e-6);
# KrylovCg!(vX, mP, mA, vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500, ρ = 1e6, σ = 1e-6);
# KrylovCr!(vX, mP, mA, vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500, ρ = 1e6, σ = 1e-6);
# KrylovCgLanczos!(vX, mP, mA, vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500, ρ = 1e6, σ = 1e-6);
# LinOpCg!(vX, mP, mA, vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500, ρ = 1e6, σ = 1e-6);
# KrylovTriCg!(vX, mP, mA, vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500, ρ = 1e6, σ = 1e-6);
# KrylovTriMr!(vX, mP, mA, vV; numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500, ρ = 1e6, σ = 1e-6);

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









