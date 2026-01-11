# ProxQP 
# A test script for the implementation 
# Minimize the Sum of Euclidean Distance to a Set of Points in 3D.
# References:
#   1.  A
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  A
# TODO:
# 	1.  AA.
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     02/01/2026  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Convex;
using ECOS;
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

include("ProxQP.jl");


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function CVXSolver( mP :: Matrix{T}, vQ :: Vector{T}, mA :: Matrix{T}, vB :: Vector{T}, mC :: Matrix{T}, vD :: Vector{T} ) where {T <: AbstractFloat}

    dataDim = size(mP, 1);

    vX = Convex.Variable(dataDim);
    sConvProb = minimize( T(0.5) * Convex.quadform(vX, mP) + Convex.dot(vQ, vX), [mA * vX == vB, mC * vX <= vD] );
    solve!(sConvProb, ECOS.Optimizer; silent = true);
    
    return vec(vX.value);
    
end


## Parameters

# Data
dataDim = 90;
numEq   = 60;
numInEq = 70;

# Solver
numIterations = 250;
η = 5e-4;


## Load / Generate Data

mP = randn(dataDim, dataDim);
mP = mP' * mP + 0.01 * I;
mP = 0.5 * (mP + mP');
vQ = randn(dataDim);

# Equality Constraints
mA = randn(numEq, dataDim);
vB = randn(numEq);
# Inequality Constraints
mC = randn(numInEq, dataDim);
vD = randn(numInEq);

## Analysis

# Reference Solution
vXRef = CVXSolver(mP, vQ, mA, vB, mC, vD);

# ProxQP Solution

sProxQP = ProxQP(mP, vQ, mA, vB, mC, vD; ρ = 25.0, σ = 1e-5);

isConv = SolveQuadraticProgram!(sProxQP; numIterations = 5_000);
println(isConv);

println(norm(sProxQP.vX - vXRef));
println(norm(mA * vXRef - vB, Inf));
println(norm(max.(mC * vXRef - vD, 0.0), Inf));


## Display Results

# figureIdx += 1;

# sTr1 = scatter3d(; x = mP[1, :], y = mP[2, :], z = mP[3, :], mode = "markers", 
#                marker_size = 7,
#                name = "Points Set", text = "Points Set");
# sTr2 = scatter3d(; x = mXi[1, :], y = mXi[2, :], z = mXi[3, :], mode = "markers", 
#                marker_size = 3,
#                name = "Optimization Path", text = "Optimization Path");
# sTr3 = scatter3d(; x = [vPRef[1]], y = [vPRef[2]], z = [vPRef[3]], mode = "markers", 
#                marker_size = 5,
#                name = "Optimal Point", text = "Optimal Point");
# sLayout = Layout(title = "Closest Point", width = 600, height = 600, 
#                  xaxis_title = "x", yaxis_title = "y",
#                  hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
#                  legend = attr(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99));

# hP = Plot([sTr1, sTr2, sTr3], sLayout);
# display(hP);

# if (exportFigures)
#     figFileNme = @sprintf("Figure%04d.png", figureIdx);
#     savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
# end


# function GenSpdMat( dataDim :: Int ) :: Matrix{Float64}

#     mA = randn(dataDim, dataDim);
#     mP = mA' * mA + 0.01 * I;
#     mP = 0.5 * (mP + mP');

#     return mP;

# end


# mAA = GenSpdMat(5);
# mBB = GenSpdMat(5);
# vBB = rand(5);

# sCA = cholesky(mAA);
# sCB = cholesky(mBB);

# vT1 = sCB \ vBB;
# vT2 = sCA \ vBB;

# println(norm(vT1 - vT2));

# cholesky!(mBB);

# copyto!(sCA.U, UpperTriangular(mBB));
# copyto!(sCA.factors, mBB);


# vT1 = sCB \ vBB;
# vT2 = sCA \ vBB;

# println(norm(vT1 - vT2));

