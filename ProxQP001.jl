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

function CVXSolver( mA :: Matrix{T}, vB :: Vector{T}, mC :: Matrix{T}, vD :: Vector{T}, mE :: Matrix{T}, vF :: Vector{T} ) where {T <: AbstractFloat}

    dataDim = size(mA, 1);

    vX = Convex.Variable(dataDim);
    sConvProb = minimize( T(0.5) * Convex.quadform(vX, mA) + Convex.dot(vB, vX), [mC * vX == vD, mE * vX <= vF] );
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

mA = randn(dataDim, dataDim);
mA = mA' * mA + 0.01 * I;
mA = 0.5 * (mA + mA');
vB = randn(dataDim);

mC = randn(numEq, dataDim);
vD = randn(numEq);

mE = randn(numInEq, dataDim);
vF = randn(numInEq);

## Analysis

# Reference Solution
vXRef = CVXSolver(mA, vB, mC, vD, mE, vF);

# ProxQP Solution

vX = zeros(dataDim);
vZ = zeros(numInEq);

isConv = SolveQuadraticProgram!(vX, vZ, mA, vB, mC, vD, mE, vF; numIterations = 5_000);
println(isConv);

println(norm(vX - vXRef))
println(norm(mC * vXRef - vD))
println(norm(max.(mE * vXRef - vF, 0.0), Inf))


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

