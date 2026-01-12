using LinearAlgebra;
# using SparseArrays;
# using IterativeSolvers;
# using MKLSparse; #<! Impportant for the Iterative Solver (Much faster sparseMat * denseVec)

# using Parameters #<! Might help with packing / unpacking of parameters

# https://docs.julialang.org/en/v1/manual/unicode-input/

# @enum LinearSolverMode modeAuto = 1 modeItertaive modeDirect
# @enum ConvergenceFlag convNumItr = 1 convAdmm convPrimDual

# Solving using ProxQP like approach
function _SolveQuadraticProgram!(vX :: Vector{T}, vZ :: Vector{T}, mA, vB, mC, vD, mE, vF;
    numIterations :: N = 2000, ϵAbs = T(1e-7), ϵRel = T(1e-6), numItrConv :: N = 10,
    ρ = T(1e-2), γ = T(1e2)) where {T <: AbstractFloat, N <: Integer}
    # Solves:
    # \aeg \min_x 0.5 * x' * A * x + b' * x
    # Subject To: C * x  = d
    #             E * x <= f
    # Solves the nearest feasible QP problem.
    
    numRowsA = size(mA, 1);
    numRowsC = size(mC, 1);
    numRowsE = size(mE, 1);

    vY = zeros(T, numRowsC);
    vS = max.(vF - mE * vX, zero(T));
    # vS = zeros(numRowsE);

    mM = mA + γ * (mC' * mC) + γ * (mE' * mE) + ρ * I;
    mM = T(0.5) * (mM' + mM);
    sC = cholesky(mM; check = false);

    convFlag = false;

    ρ¹ = inv(ρ);
    γ¹ = inv(γ);
    ϵAdmm = min(ϵAbs, ϵRel) * T(1e-2);

    vX₁ = copy(vX);
    vY₁ = copy(vY);
    vZ₁ = copy(vZ);
    
    for ii in 1:numIterations
        copyto!(vX₁, vX);
        copyto!(vY₁, vY);
        copyto!(vZ₁, vZ);

        vR = -vB - mC' * vY + γ * mC' * vD - mE' * vZ + γ * mE' * (vF - vS) + ρ * vX; #<! Right hand side vector
        copyto!(vX, sC \ vR);

        vS = max.(vF - mE * vX - γ¹ * vZ, zero(T));
        vY = vY + γ * (mC * vX - vD);
        vZ .+= γ * (mE * vX - vF + vS);
        vZ .= max.(vZ, zero(T));

        if (mod(ii, numItrConv) == 0)
            convFlag = _CheckConvergence(vX, mA, vB, mC, vD, mE, vF, vS, vY, vZ, ϵAbs, ϵRel)
        end
    end
    
    return convFlag;
    
end


function _CheckConvergence(vX :: Vector{T}, mA, vB, mC, vD, mE, vF, vS, vY, vZ, ϵAbs, ϵRel) where {T <: AbstractFloat}
    #TODO: Use buffers for the intermediate allocations (mP * vX, mA * vX, ...)
    MIN_VAL_Ρ = 1e-3;
    MAX_VAL_Ρ = 1e6;

    convFlag = false;
    
    # Pre Computation
    normResPrim = max(norm(mC * vX - vD, Inf), norm(mE * vX - vF + vS, Inf));
    normResDual = norm(mA * vX + vB + mC' * vY + mE' * vZ, Inf);
    
    maxNormPrim = max(norm(mC * vX, Inf), norm(vD, Inf), norm(mE * vX, Inf), norm(vF, Inf), norm(vS, Inf));
    maxNormDual = max(norm(mA * vX, Inf), norm(mC' * vY, Inf), norm(mE' * vZ, Inf), norm(vB, Inf));
    
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


