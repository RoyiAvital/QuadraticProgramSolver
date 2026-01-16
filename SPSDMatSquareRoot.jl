# Symmetric Positive Semi Definite Matrix Square Root
# This script check method to calculate the square root of a Symmetric Positive Definite Matrix (SPD).
# Given an SPSD matrix A, the goal is to find matrix X such that: A = X' * X.
# The matrix A must be a singular matrix.
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
# - 1.0.000     16/01/2026  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
using SparseArrays;
# External
using BenchmarkTools;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(RNG_SEED);
seedNum = Int(mod(round(time()), 5000));
oRng = StableRNG(seedNum);


## Types & Structs


## Functions

function GenMat( numRows :: N, numCols :: N, densityProb :: T, oRng :: AbstractRNG; sparseThr :: T = T(0.5) ) where {N <: Integer, T <: AbstractFloat}

    if (densityProb < sparseThr)
        mA = sprand(oRng, numRows, numCols, densityProb);
    else
        mA = randn(oRng, numRows, numCols);
    end

    return mA;

end

function CalcSPSDSquareRoot( mA :: SparseMatrixCSC{T}; valShift :: T = T(1e-10) ) where {T <: AbstractFloat}
    # Computes M such that A = M' * M for SPSD matrix A.
    # The matrix A may be a singular matrix.
    # Uses QR decomposition approach from the derivation
    
    numRows = size(mA, 1);
    
    sC = cholesky(mA; shift = valShift, check = false);

    if !issuccess(sC)
        error("The Shifted Cholesky decomposition failed. Matrix may not be SPSD.");
    end
    
    # Extract sparse L and permutation
    mL = sparse(sC.L);
    vP = sC.p;
    
    # M = L' * P' so that A = P * L * L' * P' = M' * M
    # Apply inverse permutation to rows of L'
    vPinv = invperm(vP);
    mM = mL'[:, vPinv];
    
    return mM;
    
end

function CalcSPSDSquareRoot( mA :: Matrix{T}; tolRank :: T = T(1e-10) ) where {T <: AbstractFloat}
    # Computes M such that A = M' * M for SPSD matrix A
    # Uses QR decomposition approach from the derivation
    # Based on: https://math.stackexchange.com/questions/4501568.
    
    # QR decomposition of A
    sQR = qr(mA, ColumnNorm()); # Pivoted QR for rank detection
    mQ  = Matrix(sQR.Q);
    mR  = sQR.R;
    
    # Determine numerical rank from R diagonal
    vDiagR   = abs.(diag(mR));
    rankA    = count(vDiagR .> (tolRank * maximum(vDiagR)));
    
    # Extract Q1 (first r columns of Q)
    mQ1 = mQ[:, 1:rankA];
    
    # Compute Q1' * A * Q1 (should be SPD of size r x r)
    mB = mQ1' * mA * mQ1;
    mB = Symmetric(mB); # Ensure symmetry
    
    # Cholesky decomposition: B = L * L'
    oChol = cholesky(mB);
    mL    = oChol.L;
    
    # Compute M = L' * Q1'
    mM = mL' * mQ1';
    
    return mM;
    
end


## Parameters

# Data
numRows     = 1200;
numCols     = 885;
densityProb = 0.005; #<! Below 0.5 will create Sparse Matrix
densityProb = 0.65; #<! Below 0.5 will create Sparse Matrix

valShift = 1e-10;
tolRank  = 1e-10;

## Load / Generate Data

mA = GenMat(numRows, numCols, densityProb, oRng);
mA = mA * mA'; #<! Ensure SPSD
mA = 0.5 * (mA + mA'); #<! Ensure symmetry

## Analysis

# mM = CalcSPSDSquareRoot(mA; valShift = valShift);
mM = CalcSPSDSquareRoot(mA; tolRank = tolRank);
valDiff = norm(mA - mM' * mM) / norm(mA);
@printf("Relative Difference ||A - M'M|| / ||A|| = %.6e\n", valDiff);


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

