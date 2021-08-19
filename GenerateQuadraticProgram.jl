# Unit Test for `SolveQuadraticProgram()`

using LinearAlgebra
using SparseArrays

@enum ProblemClass randomQp equalityConstrainedQp optimalControl portfolioOptimization lassoOptimization huberFitting supportVectorMachine

function GenerateRandomQP(problemClass::ProblemClass, numElements::Int64 = 1000, numConstraints::Int64 = 500)

    densityFctr = 0.15;
    α           = 1e-2;
    
    if (problemClass ∈ (randomQp, equalityConstrainedQp, optimalControl, lassoOptimization, huberFitting, supportVectorMachine))
        mM = sprandn(numElements, numElements, densityFctr);
        mP = transpose(mM) * mM + α * sparse(I, numElements, numElements);
        mA = sprandn(numConstraints, numElements, densityFctr);

        vQ = randn(numElements);
        if (problemClass == randomQp)
            vL = -rand(numConstraints);
            vU = rand(numConstraints);
        else
            vL = randn(numConstraints);
            vU = copy(vL);
        end
    elseif (problemClass == portfolioOptimization)
        mD = spdiagm(rand(numElements) * sqrt(numConstraints));
        mP = [mD spzeros(numElements, numConstraints); spzeros(numConstraints, numElements) sparse(I, numConstraints, numConstraints)];
        vQ = [randn(numElements); zeros(numConstraints)];
        mF = sprandn(numElements, numConstraints, 0.5);
        mA = [transpose(mF) sparse(-I, numConstraints, numConstraints); ones(1, numElements) spzeros(1, numConstraints); sparse(I, numElements, numElements) spzeros(numElements, numConstraints)];
        vL = [zeros(numConstraints); 1; zeros(numElements)];
        vU = [zeros(numConstraints); 1; ones(numElements)];
    end
    return mP, vQ, mA, vL, vU;
end

