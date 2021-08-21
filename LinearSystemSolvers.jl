
using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Krylov
using LinearOperators

function LaLdlt!(vX, mP, mA, vV;
    numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500,
    ρ = 1e6, σ = 1e-6)
    
    # Direct Solver using Base LDLT
    
    numElementsX            = size(vX, 1);
    numRowsP, numColsP      = size(mP);
    numRowsA, numColsA      = size(mA);
    
    ρ¹ = 1 / ρ;
    
    hDL = ldlt([mP + (σ * sparse(I, numElementsX, numElementsX)) transpose(mA); mA -ρ¹ * sparse(I, numRowsA, numRowsA)]);
    
    for ii in 1:(numFactor - 1)
        hDL = ldlt([mP + (σ * sparse(I, numElementsX, numElementsX)) transpose(mA); mA -ρ¹ * sparse(I, numRowsA, numRowsA)]);
    end
    
    for ii in 1:numIterations
        vV .= hDL \ vV;
        vX .= vV[1:numElements];
    end
    
    
end

function ItrSolCg!(vX, mP, mA, vV;
    numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500,
    ρ = 1e6, σ = 1e-6)
    
    # Iterative Solver using IterativeSolver cg!()
    
    numElementsX            = size(vX, 1);
    numRowsP, numColsP      = size(mP);
    numRowsA, numColsA      = size(mA);
    
    ρ¹ = 1 / ρ;
    
    mL = mP + (σ * sparse(I, numElementsX, numElementsX)) + (ρ * (transpose(mA) * mA));
    
    for ii in 1:(numFactor - 1)
        mL = mP + (σ * sparse(I, numElementsX, numElementsX)) + (ρ * (transpose(mA) * mA));
    end
    
    for ii in 1:numIterations
        IterativeSolvers.cg!(vX, mL, vX, abstol = ϵSolver, maxiter = numItrSolver);
    end
    
    
end


function KrylovCg!(vX, mP, mA, vV;
    numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500,
    ρ = 1e6, σ = 1e-6)
    
    # Iterative Solver using IterativeSolver cg!()
    
    numElementsX            = size(vX, 1);
    numRowsP, numColsP      = size(mP);
    numRowsA, numColsA      = size(mA);
    
    ρ¹ = 1 / ρ;
    
    mL = mP + (σ * sparse(I, numElementsX, numElementsX)) + (ρ * (transpose(mA) * mA));
    
    for ii in 1:(numFactor - 1)
        mL = mP + (σ * sparse(I, numElementsX, numElementsX)) + (ρ * (transpose(mA) * mA));
    end
    
    for ii in 1:numIterations
        vT, _ = Krylov.cg(mL, vX, atol = ϵSolver, itmax = numItrSolver);
        vX .= vT;
    end
    
    
end


function KrylovCr!(vX, mP, mA, vV;
    numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500,
    ρ = 1e6, σ = 1e-6)
    
    # Iterative Solver using IterativeSolver cg!()
    
    numElementsX            = size(vX, 1);
    numRowsP, numColsP      = size(mP);
    numRowsA, numColsA      = size(mA);
    
    ρ¹ = 1 / ρ;
    
    mL = mP + (σ * sparse(I, numElementsX, numElementsX)) + (ρ * (transpose(mA) * mA));
    
    for ii in 1:(numFactor - 1)
        mL = mP + (σ * sparse(I, numElementsX, numElementsX)) + (ρ * (transpose(mA) * mA));
    end
    
    for ii in 1:numIterations
        vT, _ = Krylov.cr(mL, vX, atol = ϵSolver, itmax = numItrSolver);
        vX .= vT;
    end
    
    
end


function KrylovCgLanczos!(vX, mP, mA, vV;
    numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500,
    ρ = 1e6, σ = 1e-6)
    
    # Iterative Solver using IterativeSolver cg!()
    
    numElementsX            = size(vX, 1);
    numRowsP, numColsP      = size(mP);
    numRowsA, numColsA      = size(mA);
    
    ρ¹ = 1 / ρ;
    
    mL = mP + (σ * sparse(I, numElementsX, numElementsX)) + (ρ * (transpose(mA) * mA));
    
    for ii in 1:(numFactor - 1)
        mL = mP + (σ * sparse(I, numElementsX, numElementsX)) + (ρ * (transpose(mA) * mA));
    end
    
    for ii in 1:numIterations
        vT, _ = Krylov.cg_lanczos(mL, vX, atol = ϵSolver, itmax = numItrSolver);
        vX .= vT;
    end
    
    
end


function LinOpCg!(vX, mP, mA, vV;
    numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500,
    ρ = 1e6, σ = 1e-6)
    
    # Iterative Solver using IterativeSolver cg!()
    
    numElementsX            = size(vX, 1);
    numRowsP, numColsP      = size(mP);
    numRowsA, numColsA      = size(mA);
    
    ρ¹ = 1 / ρ;
    
    # Define linear operator that models u <- (P + AᵀA) w.
    # We need to allocate one temporary vector.
    vT = zeros(numRowsA); #<! Buffer
    opL = LinearOperator(Float64, numElements, numElements, true, true, (vU, vW, α, β) -> begin
    mul!(vT, mA, vW);
    mul!(vU, transpose(mA), vT);
    mul!(vU, mP, vW, one(Float64), ρ);
    vU .= vU .+ (σ .* vW);
end);

for ii in 1:(numFactor - 1)
    opL = LinearOperator(Float64, numElements, numElements, true, true, (vU, vW, α, β) -> begin
        mul!(vT, mA, vW);
        mul!(vU, transpose(mA), vT);
        mul!(vU, mP, vW, one(Float64), ρ);
        vU .= vU .+ (σ .* vW);
    end);
end

for ii in 1:numIterations
    vT, _ = Krylov.cg(opL, vX, atol = ϵSolver, itmax = numItrSolver);
    vX .= vT;
end


end


function KrylovTriCg!(vX, mP, mA, vV;
    numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500,
    ρ = 1e6, σ = 1e-6)
    
    # Iterative Solver using IterativeSolver cg!()
    
    numElementsX            = size(vX, 1);
    numRowsP, numColsP      = size(mP);
    numRowsA, numColsA      = size(mA);
    
    ρ¹ = 1 / ρ;

    vVV = rand(numRowsA);

    mPI = mP + (σ * sparse(I, numElementsX, numElementsX));
    # mM = cholesky(mPI);
    mM = opCholesky(mPI);

    mN = ρ * sparse(I, numRowsA, numRowsA);

    for ii in 1:(numFactor - 1)
        mN = ρ * sparse(I, numRowsA, numRowsA);
    end
    
    for ii in 1:numIterations
        vXX, vVV, _ = tricg(transpose(mA), vX, vVV; M = mM, N = mN, atol = ϵSolver, itmax = numItrSolver);
        vX .= vXX;
    end
    
    
end


function KrylovTriMr!(vX, mP, mA, vV;
    numFactor = 3, numIterations = 50, ϵSolver = 1e-6, numItrSolver = 500,
    ρ = 1e6, σ = 1e-6)
    
    # Iterative Solver using IterativeSolver cg!()
    
    numElementsX            = size(vX, 1);
    numRowsP, numColsP      = size(mP);
    numRowsA, numColsA      = size(mA);
    
    ρ¹ = 1 / ρ;

    vVV = rand(numRowsA);

    mPI = mP + (σ * sparse(I, numElementsX, numElementsX));
    # mM = cholesky(mPI);
    mM = opCholesky(mPI);

    mN = ρ * sparse(I, numRowsA, numRowsA);

    for ii in 1:(numFactor - 1)
        mN = ρ * sparse(I, numRowsA, numRowsA);
    end
    
    for ii in 1:numIterations
        vXX, vVV, _ = trimr(transpose(mA), vX, vVV; M = mM, N = mN, atol = ϵSolver, itmax = numItrSolver);
        vX .= vXX;
    end
    
    
end

