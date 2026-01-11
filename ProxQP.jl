using LinearAlgebra;
using SparseArrays;
# using IterativeSolvers;
# using MKLSparse; #<! Impportant for the Iterative Solver (Much faster sparseMat * denseVec)

# using Parameters #<! Might help with packing / unpacking of parameters

# https://docs.julialang.org/en/v1/manual/unicode-input/

# @enum LinearSolverMode modeAuto = 1 modeItertaive modeDirect
# @enum ConvergenceFlag convNumItr = 1 convAdmm convPrimDual

struct ProxQP{T <: AbstractFloat, N <: Integer}
    vX :: Vector{T} #<! Solution vector
    mP :: Matrix{T} #<! Quadratic term
    vQ :: Vector{T} #<! Linear term
    mA :: Matrix{T} #<! Equality constraint matrix
    vB :: Vector{T} #<! Equality constraint vector
    mC :: Matrix{T} #<! Inequality constraint matrix
    vD :: Vector{T} #<! Inequality constraint vector
    vY :: Vector{T} #<! Dual variables for equality constraints
    vZ :: Vector{T} #<! Dual variables for inequality constraints
    vS :: Vector{T} #<! Slack variables for inequality constraints
    mM :: Matrix{T} #<! Pre factorized matrix for the solver
    # sC :: Union{Cholesky{T, Matrix{T}}, SparseArrays.CHOLMOD.Factor{T}} #<! Cholesky factorization of mM
    sC :: Cholesky{T, Matrix{T}} #<! Cholesky factorization of mM
    vR :: Vector{T} #<! Right hand side vector
    vX1 :: Vector{T} #<! Buffer vector `(dataDim, )` for multiplications
    vX2 :: Vector{T} #<! Buffer vector `(dataDim, )` for multiplications
    vX3 :: Vector{T} #<! Buffer vector `(dataDim, )` for multiplications
    vBb :: Vector{T} #<! Buffer vector `(numEq, )` for multiplications
    vDb :: Vector{T} #<! Buffer vector `(numInEq, )` for multiplications
    dataDim :: N    #<! Dimension of the data
    numEq   :: N    #<! Number of equality constraints
    numInEq :: N    #<! Number of inequality constraints
    ρ  :: T        #<! ADMM penalty parameter for constraints (Only to set at the end of the optimization loop)
    σ  :: T        #<! Proximal penalty parameter for the Augmented Lagrangian (Only to set at the end of the optimization loop)
    ρ¹ :: T       #<! Inverse of ADMM penalty parameter for constraints

    function ProxQP(mP :: AbstractMatrix{T}, vQ :: Vector{T}, mA :: AbstractMatrix{T}, vB :: Vector{T}, mC :: AbstractMatrix{T}, vD :: Vector{T}; ρ :: T = T(1), σ :: T = T(1)) where {T <: AbstractFloat}
        dataDim = size(mP, 1);
        numEq   = size(mA, 1);
        numInEq = size(mC, 1);
        vX = zeros(T, dataDim);
        vY = zeros(T, numEq);
        vZ = zeros(T, numInEq);
        vS = zeros(T, numInEq);
        mM = mP + ρ * (mA' * mA) + ρ * (mC' * mC) + σ * I;
        mM = T(0.5) * (mM' + mM);
        sC = cholesky(mM; check = false);
        vR = zeros(T, dataDim);
        vX1 = zeros(T, dataDim);
        vX2 = zeros(T, dataDim);
        vX3 = zeros(T, dataDim);
        vBb = zeros(T, numEq);
        vDb = zeros(T, numInEq);

        new{T, typeof(dataDim)}(vX, mP, vQ, mA, vB, mC, vD, vY, vZ, vS, mM, sC, vR, vX1, vX2, vX3, vBb, vDb, dataDim, numEq, numInEq, ρ, σ, inv(ρ));
    end
end

# Solving using ProxQP like approach
function SolveQuadraticProgram!(sQpProb :: ProxQP{T, N}; numIterations :: N = 2000, ϵAbs = T(1e-7), ϵRel = T(1e-6), numItrConv :: N = 10) where {T <: AbstractFloat, N <: Integer}
    # Solves:
    # \aeg \min_x 0.5 * x' * P * x + q' * x
    # Subject To: A * x  = b
    #             C * x <= d
    # Solves the nearest feasible QP problem.

    UpdateDecomposition!(sQpProb);

    convFlag = false;
    
    for ii in 1:numIterations

        # Update `vX`
        CalculateRhs!(sQpProb);
        UpdateX!(sQpProb);

        # Update `vS`
        UpdateS!(sQpProb);

        # Update `vY`
        UpdateY!(sQpProb);

        # Update `vZ`
        UpdateZ!(sQpProb);

        # Check seldom for convergence
        if (mod(ii, numItrConv) == 0)
            convFlag = CheckConvergence!(sQpProb, ϵAbs, ϵRel);
        end
    end
    
    return convFlag;
    
end


function CalculateRhs!(sQpProb :: ProxQP{T, N}) where {T <: AbstractFloat, N <: Integer}
    # Calculates the right hand side vector for the QP problem with no allocations
    # vR = -vQ - mA' * vY + ρ * mA' * vB - mC' * vZ + ρ * mC' * (vD - vS) + σ * vX;
    # vR = -vQ + mA' * (ρ * vB - vY) + mC' * (ρ * (vD - vS) - vZ) + σ * vX;
    ρ = sQpProb.ρ;
    σ = sQpProb.σ;
    
    @. sQpProb.vR = -sQpProb.vQ + σ * sQpProb.vX; #<! σ * vX - vQ
    @. sQpProb.vBb = ρ * sQpProb.vB - sQpProb.vY; #<! ρ * vB - vY
    mul!(sQpProb.vR, sQpProb.mA', sQpProb.vBb, T(1), T(1)); #<! mA' * (ρ * vB - vY) + vR -> vR

    @. sQpProb.vDb = ρ * (sQpProb.vD - sQpProb.vS) - sQpProb.vZ; #<! ρ * (vD - vS) - vZ
    mul!(sQpProb.vR, sQpProb.mC', sQpProb.vDb, T(1), T(1)); #<! mC' * (ρ * (vD - vS) - vZ) + vR -> vR

end

function UpdateDecomposition!(sQpProb :: ProxQP{T, N}) where {T <: AbstractFloat, N <: Integer}
    # Updates the matrix decomposition for the QP problem with no allocations
    # mM = mP + ρ * (mA' * mA) + ρ * (mC' * mC) + σ * I;
    ρ = sQpProb.ρ;
    σ = sQpProb.σ;

    sQpProb.mM .= sQpProb.mP;
    mul!(sQpProb.mM, sQpProb.mA', sQpProb.mA, ρ, T(1)); #<! ρ * (mA' * mA) + mM -> mM
    mul!(sQpProb.mM, sQpProb.mC', sQpProb.mC, ρ, T(1)); #<! ρ * (mC' * mC) + mM -> mM
    # sQpProb.mM += σ * I; #<! σ * I + mM -> mM
    @views sQpProb.mM[diagind(sQpProb.mM)] .+= σ; #<! TODO: Find non allocating implementation
    @. sQpProb.mM = T(0.5) * (sQpProb.mM' + sQpProb.mM); #<! Ensure symmetry
    cholesky!(sQpProb.mM, check = false); #<! Update the Cholesky factorization
    copyto!(sQpProb.sC.U, UpperTriangular(sQpProb.mM));
    # copyto!(sQpProb.sC.factors, sQpProb.mM);
end

function UpdateX!(sQpProb :: ProxQP{T, N}) where {T <: AbstractFloat, N <: Integer}
    # Updates the solution vector for the QP problem with no allocations
    # vX = sC \ vR;
    ldiv!(sQpProb.vX, sQpProb.sC, sQpProb.vR);
end

function UpdateS!(sQpProb :: ProxQP{T, N}) where {T <: AbstractFloat, N <: Integer}
    # Updates the slack variable vector for the QP problem with no allocations
    # vS = max.(vD - mC * vX - ρ¹ * vZ, zero(T));
    ρ¹ = sQpProb.ρ¹;

    @. sQpProb.vS = sQpProb.vD - ρ¹ * sQpProb.vZ;          #<! vD - ρ¹ * vZ -> vS
    mul!(sQpProb.vS, sQpProb.mC, sQpProb.vX, T(-1), T(1)); #<! -mC * vX + vS -> vS
    @. sQpProb.vS = max(sQpProb.vS, zero(T));              #<! max.(vS, 0) -> vS
end

function UpdateY!(sQpProb :: ProxQP{T, N}) where {T <: AbstractFloat, N <: Integer}
    # Updates the dual variable vector for equality constraints for the QP problem with no allocations
    # vY = vY + ρ * (mA * vX - vB);
    ρ = sQpProb.ρ;

    @. sQpProb.vY -= ρ * sQpProb.vB;                   #<! -ρ * vB + vY -> vY
    mul!(sQpProb.vY, sQpProb.mA, sQpProb.vX, ρ, T(1)); #<! ρ * (mA * vX) + vY -> vY
end

function UpdateZ!(sQpProb :: ProxQP{T, N}) where {T <: AbstractFloat, N <: Integer}
    # Updates the dual variable vector for inequality constraints for the QP problem with no allocations
    # vZ .+= ρ * (mC * vX - vD + vS);
    # vZ .= max.(vZ, zero(T));
    ρ = sQpProb.ρ;

    @. sQpProb.vZ += ρ * (sQpProb.vS - sQpProb.vD);    #<! ρ * (vS - vD) + vZ -> vZ
    mul!(sQpProb.vZ, sQpProb.mC, sQpProb.vX, ρ, T(1)); #<! ρ * (mC * vX) + vZ -> vZ
    @. sQpProb.vZ = max(sQpProb.vZ, zero(T));                  #<! max.(vZ, 0) -> vZ
end


function CheckConvergence!(sQpProb :: ProxQP{T, N}, ϵAbs :: T, ϵRel :: T) where {T <: AbstractFloat, N <: Integer}
    # Using the convergenmce criteria from PIQP (https://arxiv.org/abs/2304.00290, Equations 13a, 13b, 13c)
    #TODO: Use buffers for the intermediate allocations (mP * vX, mA * vX, ...)
    MIN_VAL_Ρ = 1e-3;
    MAX_VAL_Ρ = 1e6;

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
    
    # Adaptive Rho
    # if (adptΡ)
    #     numeratorVal    = normResPrim * maxNormDual;
    #     denominatorVal  = normResDual * maxNormPrim;
    #     ρρ              = clamp(ρ * sqrt(numeratorVal / denominatorVal), MIN_VAL_Ρ, MAX_VAL_Ρ);
    # end
    
    # Termination
    epsPrim = ϵAbs + ϵRel * maxNormPrim;
    epsDual = ϵAbs + ϵRel * maxNormDual;
    
    if ((normResPrim < epsPrim) && (normResDual < epsDual))
        convFlag = true;
    end
    
    return convFlag;
    
end

function _NormInfDif(vA :: Vector{T}, vB :: Vector{T}) where {T <: AbstractFloat}

    valNorm = zero(T);
    for ii in eachindex(vA)
        valNorm = max(valNorm, abs(vA[ii] - vB[ii]));
    end

    return valNorm;

end

function _NormInfDif(vA :: Vector{T}, vB :: Vector{T}, vC :: Vector{T}) where {T <: AbstractFloat}

    valNorm = zero(T);
    for ii in eachindex(vA)
        valNorm = max(valNorm, abs(vA[ii] - vB[ii] + vC[ii]));
    end

    return valNorm;

end

function _NormInfDif(vA :: Vector{T}, vB :: Vector{T}, vC :: Vector{T}, vD :: Vector{T}) where {T <: AbstractFloat}

    valNorm = zero(T);
    for ii in eachindex(vA)
        valNorm = max(valNorm, abs(vA[ii] + vB[ii] + vC[ii] + vD[ii]));
    end

    return valNorm;

end


