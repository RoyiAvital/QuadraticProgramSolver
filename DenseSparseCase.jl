# Dense Sparse Case 
# An MWE for the struggle with dense and sparse constraints in the same struct.
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
# - 1.0.000     12/01/2026  Royi Avital
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

struct SPDProblem{T <: AbstractFloat, FC <: Union{Cholesky{T}, SparseArrays.CHOLMOD.Factor{T}}}
    mA :: AbstractMatrix{T};
    vB :: Vector{T};
    mP :: AbstractMatrix{T};
    vQ :: Vector{T};
    sC :: FC;  #<! Cholesky Factorization

    function SPDProblem(mA :: AbstractMatrix{T}, vB :: Vector{T}) where {T <: AbstractFloat}

        mP = mA' * mA;     #<! Place holder
        mP = T(0.5) * (mP + mP') + T(1e-3) * I; #<! Ensure SPD
        vQ = mA' * vB;     #<! Right Hand
        sC = cholesky(mP); #<! Cholesky Factorization
        return new{T, typeof(sC)}(mA, vB, mP, vQ, sC);
        
    end
end


## Functions

function UpdateDecomposition!( sProb :: SPDProblem{T, FC}, λ :: T ) where {T <: AbstractFloat, FC <: Cholesky{T}}

    UpdateP!(sProb, λ);
    copyto!(sProb.sC.U, UpperTriangular(sProb.mP));  #<! Update Cholesky Factorization

end

function UpdateDecomposition!( sProb :: SPDProblem{T, FC}, λ :: T ) where {T <: AbstractFloat, FC <: SparseArrays.CHOLMOD.Factor{T}}

    UpdateP!(sProb, λ);
    cholesky!(sProb.sC, sProb.mP);

end

function UpdateP!(sProb :: SPDProblem{T}, λ :: T) where {T <: AbstractFloat}

    sProb.mP .= (sProb.mA' * sProb.mA) + λ * I;
    sProb.mP .+= sProb.mP';
    sProb.mP .*= T(0.5);

end

function SolveProb( sProb :: SPDProblem{T} ) where {T <: AbstractFloat}

    vX = sProb.sC \ sProb.vQ;

    return vX;

end

function GenMat( dataDim :: N, densityProb :: T, oRng :: AbstractRNG; sparseThr :: T = T(0.5) ) where {N <: Integer, T <: AbstractFloat}

    if (densityProb < sparseThr)
        mA = sprand(oRng, dataDim, dataDim, densityProb);
    else
        mA = randn(oRng, dataDim, dataDim);
    end

    return mA;

end


## Parameters

# Data
dataDim     = 9;
densityProb = 0.75; #<! Below 0.5 will create Sparse Matrix
λ = 0.95;


## Load / Generate Data

mA = GenMat(dataDim, densityProb, oRng);
vB = randn(oRng, dataDim);

## Analysis

sSpdProblem = SPDProblem(mA, vB);
UpdateDecomposition!(sSpdProblem, λ);


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

