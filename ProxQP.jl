using LinearAlgebra;
using SparseArrays;
# using MKLSparse; #<! Impportant for the Iterative Solver (Much faster sparseMat * denseVec)

const MatLike{T} = Union{Matrix{T}, SparseMatrixCSC{T, Int}}
const CholFac{T} = Union{Cholesky{T}, SparseArrays.CHOLMOD.Factor{T}}

struct ProxQP{T <: AbstractFloat, MAT <: MatLike{T}, FC <: CholFac{T}, N <: Integer}
    vX  :: Vector{T} #<! Solution vector
    mP  :: MAT #<! Quadratic term (SPD Matrix)
    vQ  :: Vector{T} #<! Linear term
    mA  :: MAT #<! Equality constraint matrix
    vB  :: Vector{T} #<! Equality constraint vector
    mC  :: MAT #<! Inequality constraint matrix
    vD  :: Vector{T} #<! Inequality constraint vector
    vY  :: Vector{T} #<! Dual variables for equality constraints
    vZ  :: Vector{T} #<! Dual variables for inequality constraints
    vS  :: Vector{T} #<! Slack variables for inequality constraints
    mK  :: MAT #<! Gram matrix of all constraint matrices `(mA' * mA + mC' * mC)`
    mM  :: MAT #<! Pre factorized matrix for the solver
    vDi :: Vector{N} #<! Indices of the diagonal elements in `mM` (only for sparse case)
    sC  :: FC #<! Cholesky factorization of mM
    vR  :: Vector{T} #<! Right hand side vector
    vX1 :: Vector{T} #<! Buffer vector `(dataDim, )` for multiplications
    vX2 :: Vector{T} #<! Buffer vector `(dataDim, )` for multiplications
    vX3 :: Vector{T} #<! Buffer vector `(dataDim, )` for multiplications
    vBb :: Vector{T} #<! Buffer vector `(numEq, )` for multiplications
    vDb :: Vector{T} #<! Buffer vector `(numInEq, )` for multiplications
    dataDim :: N    #<! Dimension of the data
    numEq   :: N    #<! Number of equality constraints
    numInEq :: N    #<! Number of inequality constraints
    # ρ  :: T        #<! ADMM penalty parameter for constraints (Only to set at the end of the optimization loop)
    # σ  :: T        #<! Proximal penalty parameter for the Augmented Lagrangian (Only to set at the end of the optimization loop)
    # ρ¹ :: T        #<! Inverse of ADMM penalty parameter for constraints

    function ProxQP(mP :: MatLike{T}, vQ :: Vector{T}, mA :: MatLike{T}, vB :: Vector{T}, mC :: MatLike{T}, vD :: Vector{T}, vX :: Vector{T}, vY :: Vector{T}, vZ :: Vector{T}, vS :: Vector{T}) where {T <: AbstractFloat}
        dataDim = size(mP, 1);
        numEq   = size(mA, 1);
        numInEq = size(mC, 1);
        mAA = mA' * mA;
        mAA = T(0.5) * (mAA' + mAA);
        mCC = mC' * mC;
        mCC = T(0.5) * (mCC' + mCC);
        mK = mAA + mCC;
        mM = mP + mK + I;
        # Check if `mM` is sparse or dense
        if isa(mM, SparseMatrixCSC{T, Int})
            # Align patterns of mP, mK, mM for faster computations
            mK = AlignSparsePattern(mM, mK); #<! Now mK has the same sparsity pattern as mM (Copy)
            mP = AlignSparsePattern(mM, mP); #<! Now mP has the same sparsity pattern as mM (Copy)
            vDi = GetNzvalDiagIdxs(mM, typeof(dataDim));
        else
            vDi = zeros(typeof(dataDim), dataDim);
        end
        mM = T(0.5) * (mM' + mM);
        sC = cholesky(mM; check = false);
        vR = zeros(T, dataDim);
        vX1 = zeros(T, dataDim);
        vX2 = zeros(T, dataDim);
        vX3 = zeros(T, dataDim);
        vBb = zeros(T, numEq);
        vDb = zeros(T, numInEq);

        new{T, typeof(mP), typeof(sC), typeof(dataDim)}(vX, mP, vQ, mA, vB, mC, vD, vY, vZ, vS, mK, mM, vDi, sC, vR, vX1, vX2, vX3, vBb, vDb, dataDim, numEq, numInEq);
    end
end

# Convenience aliases
const DenseProxQP{T, N}  = ProxQP{T, Matrix{T},               Cholesky{T},                    N}
const SparseProxQP{T, N} = ProxQP{T, SparseMatrixCSC{T, Int}, SparseArrays.CHOLMOD.Factor{T}, N}


function ProxQP(mP :: Matrix{T}, vQ :: Vector{T}, mA :: Matrix{T}, vB :: Vector{T}, mC :: Matrix{T}, vD :: Vector{T}) where {T <: AbstractFloat}
    # Initializes the ProxQP problem structure with `vX` and `vZ` as solution to the QP wit only equality constraints

    dataDim = size(mP, 1);
    numEq   = size(mA, 1);
    numInEq = size(mC, 1);

    mK = [mP mA'; mA zeros(T, numEq, numEq)];
    vR = [-vQ; vB];
        
    vK  = mK \ vR;
    vX = vK[1:dataDim];
    # vX = zeros(T, dataDim);
    vY = vK[(dataDim + 1):end];
    # vY = zeros(T, numEq);
    vS = max.(vD - mC * vX, zero(T));
    vZ = zeros(T, numInEq);

    return ProxQP(mP, vQ, mA, vB, mC, vD, vX, vY, vZ, vS);

end

function ProxQP(mP :: SparseMatrixCSC{T, Int}, vQ :: Vector{T}, mA :: SparseMatrixCSC{T, Int}, vB :: Vector{T}, mC :: SparseMatrixCSC{T, Int}, vD :: Vector{T}) where {T <: AbstractFloat}
    # Initializes the ProxQP problem structure with `vX` and `vZ` as solution to the QP wit only equality constraints

    dataDim = size(mP, 1);
    numEq   = size(mA, 1);
    numInEq = size(mC, 1);

    mK = [mP mA'; mA spzeros(T, numEq, numEq)];
    vR = [-vQ; vB];
        
    vK  = mK \ vR;
    vX = vK[1:dataDim];
    # vX = zeros(T, dataDim);
    vY = vK[(dataDim + 1):end];
    # vY = zeros(T, numEq);
    vS = max.(vD - mC * vX, zero(T));
    vZ = zeros(T, numInEq);

    return ProxQP(mP, vQ, mA, vB, mC, vD, vX, vY, vZ, vS);

end

# Solving using ProxQP like approach
function SolveQuadraticProgram!(sQpProb :: ProxQP{T}; numIterations :: N = 2000, ϵAbs = T(1e-7), ϵRel = T(1e-6), numItrConv :: N = 50, ρ :: T = T(1e2), σ :: T = T(1e-2), adptΡ :: Bool = true, τ :: T = T(10)) where {T <: AbstractFloat, N <: Integer}
    # Solves:
    # \arg \min_x 0.5 * x' * P * x + q' * x
    # Subject To: A * x  = b
    #             C * x <= d
    # Solves the nearest feasible QP problem.

    # Setting the output report
    #TODO: Check if using mutable struct is faster
    dReport :: Dict{String, Real} = Dict("Converged" => false, "Iterations" => numIterations, "ρ" => ρ, "σ" => σ, "PrimalResidual" => Inf, "DualResidual" => Inf);

    ρ¹ = inv(ρ);

    UpdateDecomposition!(sQpProb, ρ, σ);

    convFlag = false;
    
    for ii in 1:numIterations

        # Update `vX`
        CalculateRhs!(sQpProb, ρ, σ);
        UpdateX!(sQpProb);

        # Update `vS`
        UpdateS!(sQpProb, ρ¹);

        # Update `vY`
        UpdateY!(sQpProb, ρ);

        # Update `vZ`
        UpdateZ!(sQpProb, ρ);

        # Check seldom for convergence
        if (mod(ii, numItrConv) == 0)
            convFlag, normResPrim, normResDual, ρ, scaleRatio, updatedΡ = CheckConvergence!(sQpProb, ϵAbs, ϵRel, ρ, adptΡ, τ);
            dReport["PrimalResidual"] = normResPrim;
            dReport["DualResidual"]   = normResDual;
            if convFlag
                dReport["Iterations"] = ii;
                # break;
            end
            if updatedΡ
                ρ¹ = inv(ρ);
                UpdateDecomposition!(sQpProb, ρ, σ);
                # @. sQpProb.vY *= scaleRatio
                # @. sQpProb.vZ *= scaleRatio
                dReport["ρ"] = ρ;
            end
        end
    end

    dReport["Converged"] = convFlag;
    
    return dReport;
    
end

function UpdateM!(sQpProb :: ProxQP{T, MAT, FC}, ρ :: T, σ :: T) where {T <: AbstractFloat, MAT <: Matrix{T}, FC <: Cholesky{T}}
    # Updates the pre factorized matrix for the QP problem with no allocations
    # mM = mP + ρ * (mA' * mA) + ρ * (mC' * mC) + σ * I;
    @. sQpProb.mM = sQpProb.mP + ρ * sQpProb.mK;
    # sQpProb.mM += σ * I; #<! σ * I + mM -> mM
    @views sQpProb.mM[diagind(sQpProb.mM)] .+= σ; #<! TODO: Find non allocating implementation
end

function UpdateM!(sQpProb :: ProxQP{T, MAT, FC}, ρ :: T, σ :: T) where {T <: AbstractFloat, MAT <: SparseMatrixCSC{T, Int}, FC <: SparseArrays.CHOLMOD.Factor{T, Int}}
    # Updates the pre factorized matrix for the QP problem with no allocations
    # mM = mP + ρ * (mA' * mA) + ρ * (mC' * mC) + σ * I;
    @. sQpProb.mM.nzval = sQpProb.mP.nzval + ρ * sQpProb.mK.nzval;
    # sQpProb.mM += σ * I; #<! σ * I + mM -> mM
    @inbounds for kk in sQpProb.vDi
        sQpProb.mM.nzval[kk] += σ;
    end
end

function UpdateDecomposition!(sQpProb :: ProxQP{T, MAT, FC}, ρ :: T, σ :: T) where {T <: AbstractFloat, MAT <: Matrix{T}, FC <: Cholesky{T}}
    # Updates the matrix decomposition for the QP problem with no allocations
    UpdateM!(sQpProb, ρ, σ);
    cholesky!(sQpProb.mM, check = false); #<! Update the Cholesky factorization
    copyto!(sQpProb.sC.U, UpperTriangular(sQpProb.mM));
    # copyto!(sQpProb.sC.factors, sQpProb.mM);
end

function UpdateDecomposition!(sQpProb :: ProxQP{T, MAT, FC}, ρ :: T, σ :: T) where {T <: AbstractFloat, MAT <: SparseMatrixCSC{T, Int}, FC <: SparseArrays.CHOLMOD.Factor{T}}
    # Updates the matrix decomposition for the QP problem with no allocations
    # Assumes the sparse pattern does not change
    UpdateM!(sQpProb, ρ, σ);
    cholesky!(sQpProb.sC, sQpProb.mM, check = false); #<! Update the Cholesky factorization
end

function CalculateRhs!(sQpProb :: ProxQP{T}, ρ :: T, σ :: T) where {T <: AbstractFloat}
    # Calculates the right hand side vector for the QP problem with no allocations
    # vR = -vQ - mA' * vY + ρ * mA' * vB - mC' * vZ + ρ * mC' * (vD - vS) + σ * vX;
    # vR = -vQ + mA' * (ρ * vB - vY) + mC' * (ρ * (vD - vS) - vZ) + σ * vX;    
    @. sQpProb.vR = -sQpProb.vQ + σ * sQpProb.vX; #<! σ * vX - vQ
    @. sQpProb.vBb = ρ * sQpProb.vB - sQpProb.vY; #<! ρ * vB - vY
    mul!(sQpProb.vR, sQpProb.mA', sQpProb.vBb, T(1), T(1)); #<! mA' * (ρ * vB - vY) + vR -> vR

    @. sQpProb.vDb = ρ * (sQpProb.vD - sQpProb.vS) - sQpProb.vZ; #<! ρ * (vD - vS) - vZ
    mul!(sQpProb.vR, sQpProb.mC', sQpProb.vDb, T(1), T(1)); #<! mC' * (ρ * (vD - vS) - vZ) + vR -> vR

end

function UpdateX!(sQpProb :: ProxQP{T}) where {T <: AbstractFloat}
    # Updates the solution vector for the QP problem with no allocations
    # vX = sC \ vR;
    ldiv!(sQpProb.vX, sQpProb.sC, sQpProb.vR);
end

function UpdateS!(sQpProb :: ProxQP{T}, ρ¹ :: T) where {T <: AbstractFloat}
    # Updates the slack variable vector for the QP problem with no allocations
    # vS = max.(vD - mC * vX - ρ¹ * vZ, zero(T));
    @. sQpProb.vS = sQpProb.vD - ρ¹ * sQpProb.vZ;          #<! vD - ρ¹ * vZ -> vS
    mul!(sQpProb.vS, sQpProb.mC, sQpProb.vX, T(-1), T(1)); #<! -mC * vX + vS -> vS
    @. sQpProb.vS = max(sQpProb.vS, zero(T));              #<! max.(vS, 0) -> vS
end

function UpdateY!(sQpProb :: ProxQP{T}, ρ :: T) where {T <: AbstractFloat}
    # Updates the dual variable vector for equality constraints for the QP problem with no allocations
    # vY = vY + ρ * (mA * vX - vB);
    @. sQpProb.vY -= ρ * sQpProb.vB;                   #<! -ρ * vB + vY -> vY
    mul!(sQpProb.vY, sQpProb.mA, sQpProb.vX, ρ, T(1)); #<! ρ * (mA * vX) + vY -> vY
end

function UpdateZ!(sQpProb :: ProxQP{T}, ρ :: T) where {T <: AbstractFloat}
    # Updates the dual variable vector for inequality constraints for the QP problem with no allocations
    # vZ .+= ρ * (mC * vX - vD + vS);
    # vZ .= max.(vZ, zero(T));
    @. sQpProb.vZ += ρ * (sQpProb.vS - sQpProb.vD);    #<! ρ * (vS - vD) + vZ -> vZ
    mul!(sQpProb.vZ, sQpProb.mC, sQpProb.vX, ρ, T(1)); #<! ρ * (mC * vX) + vZ -> vZ
    @. sQpProb.vZ = max(sQpProb.vZ, zero(T));                  #<! max.(vZ, 0) -> vZ
end


function CheckConvergence!(sQpProb :: ProxQP{T}, ϵAbs :: T, ϵRel :: T, ρ :: T, adptΡ :: Bool, τ :: T) where {T <: AbstractFloat}
    # Using the convergenmce criteria from PIQP (https://arxiv.org/abs/2304.00290, Equations 13a, 13b, 13c)
    #TODO: Use buffers for the intermediate allocations (mP * vX, mA * vX, ...)
    MIN_VAL_Ρ = 1e-5;
    MAX_VAL_Ρ = 1e5;

    convFlag = false;
    
    # Pre Computation
    mul!(sQpProb.vX1, sQpProb.mP, sQpProb.vX, T(1), T(0)); #<! mA * vX -> vX1
    mul!(sQpProb.vX2, sQpProb.mA', sQpProb.vY, T(1), T(0)); #<! mA * vX -> vX1
    mul!(sQpProb.vX3, sQpProb.mC', sQpProb.vZ, T(1), T(0)); #<! mA * vX -> vX1
    mul!(sQpProb.vBb, sQpProb.mA, sQpProb.vX, T(1), T(0)); #<! mA * vX -> vBb
    mul!(sQpProb.vDb, sQpProb.mC, sQpProb.vX, T(1), T(0)); #<! mC * vX -> vDb
    normResPrim = max(_NormInfDif(sQpProb.vBb, sQpProb.vB), _NormInfDif(sQpProb.vDb, sQpProb.vD, sQpProb.vS));
    normResDual = _NormInfDif(sQpProb.vX1, sQpProb.vX2, sQpProb.vX3, sQpProb.vQ);
    
    maxNormPrim = max(norm(sQpProb.vBb, Inf), norm(sQpProb.vB, Inf), norm(sQpProb.vDb, Inf), norm(sQpProb.vD, Inf), norm(sQpProb.vS, Inf));
    maxNormDual = max(norm(sQpProb.vX1, Inf), norm(sQpProb.vX2, Inf), norm(sQpProb.vX3, Inf), norm(sQpProb.vQ, Inf));
    
    # @printf("Primal Residual: %0.5f, Dual Residual: %0.5f, ρ = %0.5f\n", normResPrim * maxNormDual, normResDual * maxNormPrim, ρ);
    
    # Adaptive ρ
    updatedΡ = false;
    scaleRatio = one(T);
    if (adptΡ)
        # resRatio = normResPrim / normResDual;
        resRatio = (normResPrim * maxNormDual) / (normResDual * maxNormPrim); #<! Ratio relative to potential scaling
        if (resRatio > τ) || (inv(resRatio) > τ)
            updatedΡ = true;
            ρρ = clamp(ρ * sqrt(sqrt(resRatio)), MIN_VAL_Ρ, MAX_VAL_Ρ); #<! Double square root for smoother updates (Contraction mapping towards 1)
            scaleRatio = ρ / ρρ;
            ρ = ρρ;
        end
    end
    
    # Termination
    epsPrim = ϵAbs + ϵRel * maxNormPrim;
    epsDual = ϵAbs + ϵRel * maxNormDual;
    
    if ((normResPrim < epsPrim) && (normResDual < epsDual))
        convFlag = true;
    end
    
    return convFlag, normResPrim, normResDual,ρ, scaleRatio, updatedΡ;
    
end

function _NormInfDif(vA :: Vector{T}, vB :: Vector{T}) where {T <: AbstractFloat}

    valNorm = zero(T);
    @inbounds @fastmath @simd for ii in eachindex(vA)
        valNorm = max(valNorm, abs(vA[ii] - vB[ii]));
    end

    return valNorm;

end

function _NormInfDif(vA :: Vector{T}, vB :: Vector{T}, vC :: Vector{T}) where {T <: AbstractFloat}

    valNorm = zero(T);
    @inbounds @fastmath @simd for ii in eachindex(vA)
        valNorm = max(valNorm, abs(vA[ii] - vB[ii] + vC[ii]));
    end

    return valNorm;

end

function _NormInfDif(vA :: Vector{T}, vB :: Vector{T}, vC :: Vector{T}, vD :: Vector{T}) where {T <: AbstractFloat}

    valNorm = zero(T);
    @inbounds @fastmath @simd for ii in eachindex(vA)
        valNorm = max(valNorm, abs(vA[ii] + vB[ii] + vC[ii] + vD[ii]));
    end

    return valNorm;

end

# Indices into nzval for diagonal entries (requires diagonal to exist in the pattern).
# Designate N as a type parameter to avoid allocations.
function GetNzvalDiagIdxs( mA :: SparseMatrixCSC{T, Int}, N :: Type ) where {T <: Number}
    
    numElm = min(size(mA, 1), size(mA, 2))
    vDi    = Vector{N}(undef, numElm);

    @inbounds for jj in 1:numElm
        lo = mA.colptr[jj];
        hi = mA.colptr[jj + 1] - 1;
        rows = @view mA.rowval[lo:hi];
        k = searchsortedfirst(rows, jj);
        vDi[jj] = lo + k - 1;
    end

    return vDi;
end

function AlignSparsePattern(mT :: SparseMatrixCSC{T, Int}, mA :: SparseMatrixCSC{T, Int}) where {T}
    # Creates a new sparse matrix `mB` with the same pattern as `mT` and copies values from `mA` where the patterns overlap.
    # It is assumed that the pattern of `mT` is a superset of the pattern of `mA`.
    
    mB = SparseMatrixCSC(mT.m, mT.n, copy(mT.colptr), copy(mT.rowval), zeros(T, length(mT.rowval)));

    @inbounds for jj in 1:mT.n
        loT = mT.colptr[jj];
        hiT = mT.colptr[jj + 1] - 1;
        vRow = @view mT.rowval[loT:hiT];

        for pA in mA.colptr[jj]:(mA.colptr[jj + 1] - 1)
            ii = mA.rowval[pA];
            kk = searchsortedfirst(vRow, ii);
            if (kk <= length(vRow)) && (vRow[kk] == ii)
                mB.nzval[loT + kk - 1] = mA.nzval[pA];
            end
        end
    end

    return mB;
end
