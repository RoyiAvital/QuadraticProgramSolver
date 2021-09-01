
using LinearAlgebra;
using SparseArrays;
using IterativeSolvers;
using Krylov;
using LinearOperators;
using LinearMaps;
using QDLDL;
using LDLFactorizations;
using MKLSparse;

function LaLdlInit(vX, mP, vQ, mA, ρ, ρ¹, σ, numElements, numConstraints)
    
    hDL = ldlt([mP + sparse(σ * I, numElements, numElements) mA'; mA sparse(-ρ¹ * I, numConstraints, numConstraints)]);
    vV  = zeros(numConstraints + numElements); #<! Buffer
    
    vXX = @view vV[1:numElements];
    vZZ = @view vV[(numElements + 1):end];
    
    return vXX, vZZ, [hDL, vV];
    
end

function LaLdl!(tuSolver, vXX, vZZ, vX, mP, vQ, mA, vZ, vY, ρ, ρ¹, σ, numElements, numConstraints, changedΡ)
    
    if (changedΡ)
        tuSolver[1] = ldlt([mP + sparse(σ * I, numElements, numElements) mA'; mA sparse(-ρ¹ * I, numConstraints, numConstraints)]);
    end
    
    hDL = tuSolver[1];
    vV  = tuSolver[2];
    
    @. vXX = σ * vX - vQ;
    @. vZZ = vZ - ρ¹ * vY;
    vV .= hDL \ vV; #<! Like copy!() / copyto!()
    @. vZZ = vZ + ρ¹ * (vZZ - vY);
    
    return;
    
end


function QDLdlInit(vX, mP, vQ, mA, ρ, ρ¹, σ, numElements, numConstraints)
    
    hDL = QDLDL.qdldl([mP + sparse(σ * I, numElements, numElements) mA'; mA sparse(-ρ¹ * I, numConstraints, numConstraints)]);
    vV  = zeros(numConstraints + numElements); #<! Buffer
    
    vXX = @view vV[1:numElements];
    vZZ = @view vV[(numElements + 1):end];
    
    return vXX, vZZ, [hDL, vV];
    
end

function QDLdl!(tuSolver, vXX, vZZ, vX, mP, vQ, mA, vZ, vY, ρ, ρ¹, σ, numElements, numConstraints, changedΡ)
    
    if (changedΡ)
        tuSolver[1] = QDLDL.qdldl([mP + sparse(σ * I, numElements, numElements) mA'; mA sparse(-ρ¹ * I, numConstraints, numConstraints)]);
    end
    
    hDL = tuSolver[1];
    vV  = tuSolver[2];
    
    @. vXX = σ * vX - vQ;
    @. vZZ = vZ - ρ¹ * vY;
    QDLDL.solve!(hDL, vV);
    @. vZZ = vZ + ρ¹ * (vZZ - vY);
    
    return;
    
end


function FacLdlInit(vX, mP, vQ, mA, ρ, ρ¹, σ, numElements, numConstraints)
    
    # Using LDLFactorizations
    hDL = ldl([mP + sparse(σ * I, numElements, numElements) mA'; mA sparse(-ρ¹ * I, numConstraints, numConstraints)]);
    vV  = zeros(numConstraints + numElements); #<! Buffer
    
    vXX = @view vV[1:numElements];
    vZZ = @view vV[(numElements + 1):end];
    
    return vXX, vZZ, [hDL, vV];
    
end

function FacLdl!(tuSolver, vXX, vZZ, vX, mP, vQ, mA, vZ, vY, ρ, ρ¹, σ, numElements, numConstraints, changedΡ)
    
    if (changedΡ)
        tuSolver[1] = ldl([mP + sparse(σ * I, numElements, numElements) mA'; mA sparse(-ρ¹ * I, numConstraints, numConstraints)]);
    end
    
    hDL = tuSolver[1];
    vV  = tuSolver[2];
    
    @. vXX = σ * vX - vQ;
    @. vZZ = vZ - ρ¹ * vY;
    ldiv!(hDL, vV); #<! Like copy!() / copyto!()
    @. vZZ = vZ + ρ¹ * (vZZ - vY);
    
    return;
    
end


function ItrSolCgInit(vX, mP, vQ, mA, ρ, ρ¹, σ, numElements, numConstraints)
    
    mAA = mA' * mA;
    mPI = mP + sparse(σ * I, numElements, numElements);
    mL  = mPI + ρ * mAA;
    vT  = zeros(numElements); #<! Buffer
    
    vXX = zeros(numElements);
    vZZ = zeros(numConstraints);
    
    return vXX, vZZ, [mL, mPI, mAA, vT];
    
    
end

function ItrSolCg!(tuSolver, vXX, vZZ, vX, mP, vQ, mA, vZ, vY, ρ, ρ¹, σ, numElements, numConstraints, changedΡ; ϵPcg = 1e-6, numItrPcg = 1000)
    
    if (changedΡ)
        tuSolver[1] = tuSolver[2] + ρ * tuSolver[3];
    end
    
    mL = tuSolver[1];
    vT = tuSolver[4];
    
    @. vZZ = ρ * vZ - vY; #<! Using `vZZ` as a buffer
    mul!(vT, mA', vZZ);
    @. vT = σ * vX - vQ + vT;
    IterativeSolvers.cg!(vXX, mL, vT, abstol = ϵPcg, maxiter = numItrPcg);
    # cg!(vXX, mL, σ * vX - vQ + mA' * (ρ * vZ - vY), abstol = ϵPcg, maxiter = numItrPcg);
    mul!(vZZ, mA, vXX);
    
    
end


function LinOpCgInit(vX, mP, vQ, mA, ρ, ρ¹, σ, numElements, numConstraints)

    vXX = zeros(numElements);
    vZZ = zeros(numConstraints);
    
    vT  = zeros(numElements); #<! Buffer
    
    mL = LinearOperator(Float64, numElements, numElements, true, true, (vU, vW, α, β) -> begin
        mul!(vZZ, mA, vW);
        mul!(vU, mA', vZZ);
        mul!(vU, mP, vW, one(Float64), ρ);
        vU .= vU .+ (σ .* vW);
    end);

    return vXX, vZZ, [mL, vT];


end

function LinOpCg!(tuSolver, vXX, vZZ, vX, mP, vQ, mA, vZ, vY, ρ, ρ¹, σ, numElements, numConstraints, changedΡ; ϵPcg = 1e-6, numItrPcg = 1000)
    
    if (changedΡ)
        tuSolver[1] = LinearOperator(Float64, numElements, numElements, true, true, (vU, vW, α, β) -> begin
            mul!(vZZ, mA, vW);
            mul!(vU, mA', vZZ);
            mul!(vU, mP, vW, one(Float64), ρ);
            vU .= vU .+ (σ .* vW);
        end);
    end

    mL = tuSolver[1];
    vT = tuSolver[2];

    @. vZZ = ρ * vZ - vY; #<! Using `vZZ` as a buffer
    mul!(vT, mA', vZZ);
    @. vT = σ * vX - vQ + vT;
    IterativeSolvers.cg!(vXX, mL, vT, abstol = ϵPcg, maxiter = numItrPcg);
    # cg!(vXX, mL, σ * vX - vQ + mA' * (ρ * vZ - vY), abstol = ϵPcg, maxiter = numItrPcg);
    mul!(vZZ, mA, vXX);


end

function LinMapsCgInit(vX, mP, vQ, mA, ρ, ρ¹, σ, numElements, numConstraints)

    vXX = zeros(numElements);
    vZZ = zeros(numConstraints);
    
    vT  = zeros(numElements); #<! Buffer
    
    mL = LinearMap{Float64}(((vU, vW) -> begin
        mul!(vZZ, mA, vW);
        mul!(vU, mA', vZZ);
        mul!(vU, mP, vW, one(Float64), ρ);
        vU .= vU .+ (σ .* vW);
    end), numElements, numElements; issymmetric = true, isposdef = true, ismutating = true);

    return vXX, vZZ, [mL, vT];


end

function LinMapsCg!(tuSolver, vXX, vZZ, vX, mP, vQ, mA, vZ, vY, ρ, ρ¹, σ, numElements, numConstraints, changedΡ; ϵPcg = 1e-6, numItrPcg = 1000)
    
    if (changedΡ)
        tuSolver[1] = LinearMap{Float64}(((vU, vW) -> begin
            mul!(vZZ, mA, vW);
            mul!(vU, mA', vZZ);
            mul!(vU, mP, vW, one(Float64), ρ);
            vU .= vU .+ (σ .* vW);
        end), numElements, numElements; issymmetric = true, isposdef = true, ismutating = true);
    end

    mL = tuSolver[1];
    vT = tuSolver[2];

    @. vZZ = ρ * vZ - vY; #<! Using `vZZ` as a buffer
    mul!(vT, mA', vZZ);
    @. vT = σ * vX - vQ + vT;
    IterativeSolvers.cg!(vXX, mL, vT, abstol = ϵPcg, maxiter = numItrPcg);
    # cg!(vXX, mL, σ * vX - vQ + mA' * (ρ * vZ - vY), abstol = ϵPcg, maxiter = numItrPcg);
    mul!(vZZ, mA, vXX);


end

