
using LinearAlgebra;
using SparseArrays;
using IterativeSolvers;
using MKLSparse; #<! Impportant for the Iterative Solver (Much faster sparseMat * denseVec)

# using Parameters #<! Might help with packing / unpacking of parameters

# https://docs.julialang.org/en/v1/manual/unicode-input/

@enum LinearSolverMode modeAuto = 1 modeItertaive modeDirect
@enum ConvergenceFlag convNumItr = 1 convAdmm convPrimDual

function SolveQuadraticProgram!(vX, mP, vQ, mA, vL, vU, LinSysSolInit, LinSysSol!;
    numIterations = 5000, ϵAbs = 1e-6, ϵRel = 1e-6,
    ρ = 1e6, σ = 1e-6, α = 1.6, δ = 1e-6, adptΡ::Bool = false, 
    fctrΡ = 5, numItrConv = 25, numItrPolish = 10, ϵMinres = 1e-6, numItrMinres = 500)
    
    numElementsX            = size(vX, 1);
    numRowsP, numColsP      = size(mP);
    numElementsQ            = size(vQ, 1);
    numRowsA, numColsA      = size(mA);
    numElementsL            = size(vL, 1);
    numElementsU            = size(vU, 1);
    
    # TODO: Numbers should be optimized
    MAX_NUM_ROWS_L  = 5000;
    MAX_DENSITY     = 0.4;
    
    ρ¹ = 1 / ρ;
    α¹ = 1 - α;
    
    convFlag    = convNumItr;
    ϵAdmm       = min(ϵAbs, ϵRel) * 1e-2;
    
    vXX, vZZ, tuSolver = LinSysSolInit(vX, mP, vQ, mA, ρ, ρ¹, σ, numElementsX, numRowsA);
    
    vXP = zeros(numElementsX); #<! Previous iteration of vX
    vZ  = zeros(numRowsA);
    vY  = zeros(numRowsA);
    vZP = zeros(numRowsA); #<! Previous iteration of vZ
    
    ρρ = ρ;
    
    for ii in 1:numIterations
        changedΡ = false;
        if (adptΡ && ((ρρ * fctrΡ < ρ) || (ρρ > fctrΡ * ρ)))
            ρ   = ρρ;
            ρ¹  = 1 / ρ;
            
            changedΡ = true;
        end
        
        LinSysSol!(tuSolver, vXX, vZZ, vX, mP, vQ, mA, vZ, vY, ρ, ρ¹, σ, numElementsX, numRowsA, changedΡ);
        
        copyto!(vXP, vX);
        @. vX = α * vXX + α¹ * vX;
        
        copyto!(vZP, vZ);
        @. vZ = clamp(α * vZZ + α¹ * vZ + ρ¹ * vY, vL, vU); #<! Projection
        @. vY = vY + ρ * (α * vZZ + α¹ * vZP - vZ);
        
        if (mod(ii, numItrConv) == 0)
            ρρ, convFlag = CheckConvergence(vX, mP, vQ, mA, vZ, vY, vXP, vZP, ρ, ρρ, adptΡ, ϵAbs, ϵRel, ϵAdmm, convFlag);
            
            if(convFlag ≠ convNumItr)
                break;
            end
        end
        
    end
    
    return convFlag;
    
    
end


function CheckConvergence(vX, mP, vQ, mA, vZ, vY, vXP, vZP, ρ, ρρ, adptΡ, ϵAbs, ϵRel, ϵAdmm, convFlag)
    #TODO: Use buffers for the intermediate allocations (mP * vX, mA * vX, ...)
    MIN_VAL_Ρ = 1e-3;
    MAX_VAL_Ρ = 1e6;
    
    # Pre Computation
    normResPrim = norm(mA * vX - vZ, Inf);
    normResDual = norm(mP * vX + vQ + mA' * vY, Inf);
    
    maxNormPrim = max(norm(mA * vX, Inf), norm(vZ, Inf));
    maxNormDual = max(norm(mP * vX, Inf), norm(mA' * vY, Inf), norm(vQ, Inf));
    
    # Adaptive Rho
    if (adptΡ)
        numeratorVal    = normResPrim * maxNormDual;
        denominatorVal  = normResDual * maxNormPrim;
        ρρ              = clamp(ρ * sqrt(numeratorVal / denominatorVal), MIN_VAL_Ρ, MAX_VAL_Ρ);
    end
    
    # Termination
    epsPrim = ϵAbs + ϵRel * maxNormPrim;
    epsDual = ϵAbs + ϵRel * maxNormDual;
    
    if ((normResPrim < epsPrim) && (normResDual < epsDual))
        convFlag = convPrimDual;
    end
    if ((norm(vX - vXP, Inf) <= ϵAdmm) && (norm(vZ - vZP, Inf) <= ϵAdmm))
        convFlag = convAdmm;
    end
    
    return ρρ, convFlag;
    
    
end


# Left for reference
function SolveQuadraticProgramRef!(vX, mP, vQ, mA, vL, vU;
    numIterations = 5000, ϵAbs = 1e-6, ϵRel = 1e-6,
    ρ = 1e6, σ = 1e-6, α = 1.6, δ = 1e-6, adptΡ::Bool = false, 
    fctrΡ = 5, numItrPolish = 10, linSolverMode::LinearSolverMode = modeAuto,
    ϵPcg = 1e-9, numItrPcg = 15000, ϵMinres = 1e-6, numItrMinres = 500, numItrConv = 25)
    
    numElementsX            = size(vX, 1);
    numRowsP, numColsP      = size(mP);
    numElementsQ            = size(vQ, 1);
    numRowsA, numColsA      = size(mA);
    numElementsL            = size(vL, 1);
    numElementsU            = size(vU, 1);
    
    # TODO: Numbers should be optimized
    MAX_NUM_ROWS_L  = 5000;
    MAX_DENSITY     = 0.4;
    
    MIN_VAL_Ρ = 1e-3;
    MAX_VAL_Ρ = 1e6;
    
    ρ¹ = 1 / ρ;
    α¹ = 1 - α;
    
    if (linSolverMode == modeItertaive)
        directSol = false;
    elseif (linSolverMode == modeDirect)
        directSol = true;
    else
        numRowsL    = numRowsP + numRowsA;
        numNonZeros = nnz(mP) + nnz(mA);
        nnzDensity  = numNonZeros / (numRowsL * numRowsL);
        if ((numRowsL <= MAX_NUM_ROWS_L) && (nnzDensity <= MAX_DENSITY))
            directSol = true;
        else
            directSol = false;
        end
    end
    
    convFlag    = convNumItr;
    ϵAdmm       = min(ϵAbs, ϵRel) * 1e-2;
    
    vXX = copy(vX);
    vXP = zeros(numElementsX); #<! Previous iteration of vX
    vZ  = zeros(numRowsA);
    vY  = zeros(numRowsA);
    vZZ = zeros(numRowsA);
    vZP = zeros(numRowsA); #<! Previous iteration of vZ
    
    vV  = zeros(numElementsX + numRowsA);
    vT1 = @view vV[1:numElementsX];
    vT2 = @view vV[(numElementsX + 1):end];
    
    if (adptΡ)
        # ̂ρ = ρ; #<! Won't work
        ρρ = ρ;
        
        mAA = mA' * mA;
        mPI = mP + sparse(σ * I, numElementsX, numElementsX);
    end
    
    if (directSol)
        hDL = ldlt([mP + sparse(σ * I, numElementsX, numElementsX) mA'; mA sparse(-ρ¹ * I, numRowsA, numRowsA)]);
    else
        mL = mP + sparse(σ * I, numElementsX, numElementsX) + (ρ * (mA' * mA));
        # sCgSolver = Krylov.CgSolver(numElementsX, numElementsX, typeof(vX)); #<! For Krylov based
        # mL = LinearOperator(Float64, numElements, numElements, true, true, (vU, vW, α, β) -> begin
        #         mul!(vU, mAA, vW);
        #         mul!(vU, mPI, vW, one(Float64), ρ);
        #     end);
    end
    
    for ii in 1:numIterations
        if (adptΡ && ((ρρ * fctrΡ < ρ) || (ρρ > fctrΡ * ρ)))
            ρ   = ρρ;
            ρ¹  = 1 / ρ;
            if (directSol)
                hDL = ldlt([mP + sparse(σ * I, numElementsX, numElementsX) mA'; mA sparse(-ρ¹ * I, numRowsA, numRowsA)]);
            else
                mL = mPI + (ρ * mAA);
                #     mL = LinearOperator(Float64, numElements, numElements, true, true, (vU, vW, α, β) -> begin
                #     mul!(vU, mAA, vW);
                #     mul!(vU, mPI, vW, one(Float64), ρ);
                # end);
            end
        end
        if (directSol)
            # vV = hDL \ [σ * vX - vQ; vZ - ρ¹ * vY];
            # @. vXX = vV[1:numElementsX];
            # @. vZZ = vZ + ρ¹ * (vV[(numElementsX + 1):end] - vY);
            @. vT1 = σ * vX - vQ;
            @. vT2 = vZ - ρ¹ * vY;
            # ldiv!(hDL, vV); #<! No support in SuiteSparse
            vV .= hDL \ vV; #<! Like copy!() / copyto!()
            @. vXX = vV[1:numElementsX];
            @. vZZ = vZ + ρ¹ * (vT2 - vY);
        else
            @. vT2 = ρ * vZ - vY;
            mul!(vT1, mA', vT2);
            @. vT1 = σ * vX - vQ + vT1;
            IterativeSolvers.cg!(vXX, mL, vT1, abstol = ϵPcg, maxiter = numItrPcg);
            # cg!(vXX, mL, σ * vX - vQ + mA' * (ρ * vZ - vY), abstol = ϵPcg, maxiter = numItrPcg);
            mul!(vZZ, mA, vXX);
        end
        
        copyto!(vXP, vX);
        @. vX = α * vXX + α¹ * vX;
        
        copyto!(vZP, vZ);
        @. vZ = clamp(α * vZZ + α¹ * vZ + ρ¹ * vY, vL, vU); #<! Projection
        @. vY = vY + ρ * (α * vZZ + α¹ * vZP - vZ);
        
        if (mod(ii, numItrConv) == 0)
            # Pre Computation
            normResPrim = norm(mA * vX - vZ, Inf);
            normResDual = norm(mP * vX + vQ + mA' * vY, Inf);
            
            maxNormPrim = max(norm(mA * vX, Inf), norm(vZ, Inf));
            maxNormDual = max(norm(mP * vX, Inf), norm(mA' * vY, Inf), norm(vQ, Inf));
            
            # Adaptive Rho
            if (adptΡ)
                numeratorVal    = normResPrim * maxNormDual;
                denominatorVal  = normResDual * maxNormPrim;
                ρρ              = clamp(ρ * sqrt(numeratorVal / denominatorVal), MIN_VAL_Ρ, MAX_VAL_Ρ);
            end
            
            # Termination
            epsPrim = ϵAbs + ϵRel * maxNormPrim;
            epsDual = ϵAbs + ϵRel * maxNormDual;
            
            if ((normResPrim < epsPrim) && (normResDual < epsDual))
                convFlag = convPrimDual;
                break;
            end
            if ((norm(vX - vXP, Inf) <= ϵAdmm) && (norm(vZ - vZP, Inf) <= ϵAdmm))
                convFlag = convAdmm;
                break;
            end
        end
        
    end
    
    return convFlag;
    
    
end