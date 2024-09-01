# Generates Symmetric Sparse Matrices

using MatrixDepot;
using SparseArrays;

@enum MatrixType randomQp = 1 randomPoisson randomTriDiag weightedLs

include("GenerateQuadraticProgram.jl")

function GenerateSpraseSymMatrix(matrixType::MatrixType; numRows::Integer = 100)
    
    if (matrixType == randomQp)
        mA, _ = GenerateRandomQP(randomQp, numRows);
    elseif (matrixType == randomPoisson)
        mA = matrixdepot("poisson", ceil(typeof(numRows), sqrt(numRows)));
    elseif (matrixType == randomTriDiag)
        mA = matrixdepot("tridiag", numRows);
    elseif (matrixType == weightedLs) #TODO: Copy from my MATLAB code
        mA = matrixdepot("tridiag", numRows);


end


function ConvertSparseMatrixIndType(mA, indType; valType = Float64)
    
    vI, vJ, vK = findnz(mA);
    return sparse(indType.(vI), indType.(vJ), vK, mA.m, mA.n);

    # Other options
    # SparseMatrixCSC{valType, indType}(mA.m, mA.n, indType.(mA.colptr), indType.(mA.rowval), valType.(mA.nzval));
    # convert(SparseMatrixCSC{valType, indType}, mA);


end