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
# - 1.0.000     13/01/2026  Royi Avital
#   *   First release.

## Packages

# Internal
using DelimitedFiles;
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

oRng = StableRNG(RNG_SEED);
seedNum = Int(mod(round(time()), 5000));
oRng = StableRNG(seedNum);
# oRng = StableRNG(1220);


## Functions

function CVXSolver( mP :: MAT, vQ :: Vector{T}, mA :: MAT, vB :: Vector{T}, mC :: MAT, vD :: Vector{T} ) where {T <: AbstractFloat, MAT <: Union{Matrix{T}, SparseMatrixCSC{T}}}

    dataDim = size(mP, 1);

    sC = cholesky(mP; check = false, perm = 1:dataDim);
    mU = sparse(sparse(sC.L)');

    vX = Convex.Variable(dataDim);
    # sConvProb = minimize( T(0.5) * Convex.quadform(vX, mP; assume_psd = true) + Convex.dot(vQ, vX), [mA * vX == vB, mC * vX <= vD] ); #<! See https://github.com/jump-dev/Convex.jl/issues/725
    sConvProb = minimize( T(0.5) * Convex.square(Convex.norm2(mU * vX)) + Convex.dot(vQ, vX), [mA * vX == vB, mC * vX <= vD] ); #<! See https://github.com/jump-dev/Convex.jl/issues/725
    solve!(sConvProb, ECOS.Optimizer; silent = true);
    
    return vec(vX.value);
    
end

function GenDiffOp( diffPow :: Int, numSamples :: Int )

    dCoeff = Dict(
        1 => [-0.5, 0.0, 0.5],
        2 => [1.0, -2.0, 1.0],
        3 => [-0.5, 1.0, 0.0, -1.0, 0.5],
        4 => [1.0, -4.0, 6.0, -4.0, 1.0],
        5 => [-0.5, 2.0, -2.5, 0.0, 2.5, -2.0, 0.5],
        6 => [1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0],
    )

    vC = dCoeff[diffPow];
    numCoeff = length(vC);
    coeffRadius = ((numCoeff - 1) ÷ 2);

    mD = spdiagm(numSamples, numSamples, (ii => vC[ii + coeffRadius + 1] * ones(numSamples - abs(ii)) for ii in -coeffRadius:coeffRadius)...);
    mD = mD[(coeffRadius + 1):(end - coeffRadius), :];

    return mD;

end

function GenMonoOp( vI :: Vector{Int}, vY :: Vector{T} ) where {T <: AbstractFloat}
    # Assumes `vI` is sorted

    numRefPts = length(vI);
    numSamples = length(vY);
    
    numSegments = numRefPts - 1;

    # Default Monotonic Non Decreasing
    # Forcing the monotonicity by the relationship with the next sample
    mD = ones(T, numSamples, 2); #<! Current sample
    mD[:, 2] .= T(-1); #<! Next sample

    for ii = 1:numSegments
        startIdx = vI[ii];
        endIdx   = vI[ii + 1];
        
        if vY[startIdx] <= vY[endIdx]
            # Monotonic Non Decreasing
            valSign = one(T);
        else
            # Monotonic Non Increasing
            valSign = -one(T);
        end
        
        for jj = startIdx:(endIdx - 1)
            mD[jj, :] .*= valSign;
        end
    
    end

    # https://discourse.julialang.org/t/39604
    mA = spdiagm(numSamples, numSamples, 0 => mD[:, 1], 1 => mD[1:(end - 1), 2]); 
    mA = mA[vI[1]:(vI[end] - 1), :];

    return mA;

end


## Parameters

# Data
csvFileName = "exchange_rate.csv"; #<! https://huggingface.co/datasets/thuml/Time-Series-Library/tree/main/exchange_rate
varIdx      = 2;
decFactor   = 50;

# Model
diffPow       = 2;
paramLambda   = 1.95;
numRefPtsFctr = 0.025;


## Load / Generate Data

# Read the CSV Data
mData = readdlm(csvFileName, ',', skipstart = 1);
vY = Float64.(mData[:, varIdx]);
vY = vY[1:decFactor:end];

numSamples = length(vY);
vT = 1:numSamples;

mD = GenDiffOp(diffPow, numSamples);

numRefPts = Int(round(numRefPtsFctr * numSamples));
vI = sort(randperm(numSamples)[1:numRefPts]); #<! Not efficient

mP = I + paramLambda * (mD' * mD);
vQ = -vY;

mA = sparse(1:numRefPts, vI, 1.0, numRefPts, numSamples); #<! Equality Constraints
vB = vY[vI];
mC = GenMonoOp(vI, vY); #<! Inequality Constraints
# mC.nzval .= 0.0; #<! Disable Monotonicity Constraints
vD = zeros(size(mC, 1));


## Analysis

# Reference Solution
vXRef = CVXSolver(mP, vQ, mA, vB, mC, vD);

# ProxQP Solution
sProxQP = ProxQP(mP, vQ, mA, vB, mC, vD);

dReport = SolveQuadraticProgram!(sProxQP; numIterations = 5_000, ρ = 200.0, σ = 1e-2, adptΡ = true, τ = 10.0);
isConv = dReport["Converged"];
println(isConv);

vX = copy(sProxQP.vX);

println(norm(sProxQP.vX - vXRef));
println(norm(mA * vXRef - vB, Inf));
println(norm(max.(mC * vXRef - vD, 0.0), Inf));


## Display Results

figureIdx += 1;

sTr1 = scatter(; x = vT, y = vY, mode = "lines", 
               line_width = 3,
               name = "Data", text = "Data");
sTr2 = scatter(; x = vT[vI], y = vY[vI], mode = "markers", 
               marker_size = 10,
               name = "Reference Points", text = "Reference Points");
sTr3 = scatter(; x = vT, y = vXRef, mode = "lines", 
               line_width = 3,
               name = "Spline Curve", text = "Spline Curve");
sLayout = Layout(title = "Spline Piece Wise Monotonic Smooth", width = 600, height = 600, 
                 xaxis_title = "x", yaxis_title = "y",
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                 legend = attr(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99));

hP = Plot([sTr1, sTr2, sTr3], sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end


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

# Benchmamark the SolveQuadraticProgram!()

# @btime SolveQuadraticProgram!(sProxQP; numIterations = 5_000, ρ = 200.0, σ = 1e-2, adptΡ = true, τ = 10.0) setup = (sProxQP = ProxQP(mP, vQ, mA, vB, mC, vD));




# using ECOS;
# using SparseArrays;

# function SolveQuadForm( mP :: AbstractMatrix{T}, vQ :: Vector{T} ) where {T <: AbstractFloat}

#     dataDim = size(mP, 1);

#     sC = cholesky(mP; check = false, perm = 1:dataDim);
#     mU = sparse(sparse(sC.L)');

#     vX = Convex.Variable(dataDim);
#     # sConvProb = minimize( T(0.5) * Convex.quadform(vX, mP; assume_psd = true) + Convex.dot(vQ, vX) );
#     sConvProb = minimize( T(0.5) * Convex.square(Convex.norm2(mU * vX)) + Convex.dot(vQ, vX) );
#     solve!(sConvProb, ECOS.Optimizer; silent = true);
    
#     return vec(vX.value);
    
# end

# dataDim = 10;

# mP = sprand(dataDim, dataDim, 0.5);
# mP = (mP' * mP) + 0.5 * I;
# mP = 0.5 * (mP + mP');
# vQ = randn(dataDim);

# vX = SolveQuadForm(mP, vQ);

# dataDim = 10;

# mP = sprand(dataDim, dataDim, 0.5);
# mP = (mP' * mP) + 0.5 * I;
# mP = 0.5 * (mP + mP');
# vQ = randn(dataDim);
# vX = zeros(dataDim);

# sC = cholesky(mP; check = false);
# ldiv!(vX, sC, vQ);