# Unit Test for `SolveQuadraticProgram()`

using LinearAlgebra
using SparseArrays

@enum ProblemClass randomQp = 1 inequalityConstrainedQp equalityConstrainedQp optimalControl portfolioOptimization lassoOptimization huberFitting supportVectorMachine isotonicRegression

function GenerateRandomQP(problemClass::ProblemClass, numElements::Int64 = 1000, numConstraints::Int64 = 500)
    
    if (problemClass ∈ (randomQp, inequalityConstrainedQp, equalityConstrainedQp, optimalControl, lassoOptimization, huberFitting, supportVectorMachine))
        densityFctr = 0.15;
        α           = 1e-2;
        
        mM = sprandn(numElements, numElements, densityFctr);
        mP = transpose(mM) * mM + sparse(α * I, numElements, numElements);
        mA = sprandn(numConstraints, numElements, densityFctr);
        
        vQ = randn(numElements);
        if (problemClass == inequalityConstrainedQp)
            vL = -rand(numConstraints);
            vU = rand(numConstraints);
        elseif (problemClass == equalityConstrainedQp)
            vL = randn(numConstraints);
            vU = copy(vL);
        else
            vL = -rand(numConstraints);
            vU = rand(numConstraints);
            vI = rand(numConstraints) .<= 0.15;
            vL[vI] .= vU[vI];
            vI = rand(numConstraints) .<= 0.15;
            vU[vI] .= vI[vI];
        end
    elseif (problemClass == portfolioOptimization)
        mD = spdiagm(rand(numElements) * sqrt(numConstraints));
        mP = [mD spzeros(numElements, numConstraints); spzeros(numConstraints, numElements) sparse(I, numConstraints, numConstraints)];
        vQ = [randn(numElements); zeros(numConstraints)];
        mF = sprandn(numElements, numConstraints, 0.5);
        mA = [transpose(mF) sparse(-I, numConstraints, numConstraints); ones(1, numElements) spzeros(1, numConstraints); sparse(I, numElements, numElements) spzeros(numElements, numConstraints)];
        vL = [zeros(numConstraints); 1; zeros(numElements)];
        vU = [zeros(numConstraints); 1; ones(numElements)];
    elseif (problemClass == isotonicRegression)
        densityFctr = 0.25;
        α           = 1e-2;

        mM = sprandn(numElements, numElements, densityFctr);
        mP = transpose(mM) * mM + sparse(α * I, numElements, numElements);
        vQ = randn(numElements);
        
        if (rand() >= 0.5)
            # Monotonic Non Increasing
            mA = spdiagm(numElements - 1, numElements, 0 => ones(numElements), 1 => -ones(numElements));
        else
            # Monotonic Non Decreasing
            mA = spdiagm(numElements - 1, numElements, 0 => -ones(numElements), 1 => ones(numElements));
        end
        vL = zeros(numElements - 1);
        vU = 10 * ones(numElements - 1);
    end

    return mP, vQ, mA, vL, vU;


end

