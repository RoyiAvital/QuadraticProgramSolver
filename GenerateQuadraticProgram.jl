# Unit Test for `SolveQuadraticProgram()`

using LinearAlgebra
using SparseArrays

@enum ProblemClass randomQp = 1 inequalityConstrainedQp equalityConstrainedQp optimalControl portfolioOptimization lassoOptimization huberFitting supportVectorMachine isotonicRegression

function GenerateRandomQP(problemClass::ProblemClass, numElements::Int64 = 1000; numConstraints::Int64 = 0)
    
    if (problemClass ∈ (randomQp, inequalityConstrainedQp, equalityConstrainedQp, optimalControl))
        densityFctr = 0.15;
        α           = 1e-2;

        mM = sprandn(numElements, numElements, densityFctr);
        mP = (mM' * mM) + sparse(α * I, numElements, numElements);        
        vQ = randn(numElements);
        if (problemClass == inequalityConstrainedQp)
            numConstraints ≠ 0 || (numConstraints = 10 * numElements);
            mA = sprandn(numConstraints, numElements, densityFctr);
            vL = -rand(numConstraints);
            vU = rand(numConstraints);
        elseif (problemClass == equalityConstrainedQp)
            numConstraints ≠ 0 || (numConstraints = numElements ÷ 2);
            mA = sprandn(numConstraints, numElements, densityFctr);
            vL = randn(numConstraints);
            vU = copy(vL);
        else
            numConstraints ≠ 0 || (numConstraints = numElements ÷ 2);
            mA = sprandn(numConstraints, numElements, densityFctr);
            vL = -rand(numConstraints);
            vU = rand(numConstraints);
            vI = rand(numConstraints) .<= 0.15;
            vL[vI] .= vU[vI];
            vI = rand(numConstraints) .<= 0.15;
            vU[vI] .= vI[vI];
        end
    elseif (problemClass == portfolioOptimization)
        densityFctr = 0.5;

        numConstraints ≠ 0 || (numConstraints = max(5, numElements ÷ 100));
        mD = spdiagm(rand(numElements) * sqrt(numConstraints));
        mP = [mD spzeros(numElements, numConstraints); spzeros(numConstraints, numElements) sparse(I, numConstraints, numConstraints)];
        vQ = [randn(numElements); zeros(numConstraints)];
        mF = sprandn(numElements, numConstraints, densityFctr);
        mA = [mF' sparse(-I, numConstraints, numConstraints); ones(1, numElements) spzeros(1, numConstraints); sparse(I, numElements, numElements) spzeros(numElements, numConstraints)];
        vL = [zeros(numConstraints); 1; zeros(numElements)];
        vU = [zeros(numConstraints); 1; ones(numElements)];
    elseif (problemClass == lassoOptimization)
        densityFctr = 0.15;

        numConstraints ≠ 0 || (numConstraints = numElements * 100);
        mAd = sprandn(numConstraints, numElements, densityFctr);
        vXX = (randn(numElements) ./ sqrt(numElements)) .* (rand(numElements) .> 0.5);
        vB  = mAd * vXX + randn(numConstraints);
        λ   = norm(mAd' * vB, Inf) / 5.0;

        mP = blockdiag(spzeros(numElements, numElements), sparse(2I, numConstraints, numConstraints), spzeros(numElements, numElements));
        vQ = [zeros(numElements + numConstraints); λ .* ones(numElements)];
        mA = [mAd sparse(-I, numConstraints, numConstraints) spzeros(numConstraints, numElements); sparse(I, numElements, numElements) spzeros(numElements, numConstraints) sparse(-I, numElements, numElements); sparse(I, numElements, numElements) spzeros(numElements, numConstraints) sparse(I, numElements, numElements)];
        vL = [vB; -Inf .* ones(numElements); zeros(numElements)];
        vU = [vB; zeros(numElements); Inf .* ones(numElements)];
    elseif (problemClass == huberFitting)
        densityFctr = 0.15;

        numConstraints ≠ 0 || (numConstraints = numElements * 100);
        mAd = sprandn(numConstraints, numElements, densityFctr);
        vXX = randn(numElements) ./ sqrt(numElements);
        vI = rand(numConstraints) .< 0.95;
        vB = (mAd * vXX) + (0.5 .* vI .* randn(numConstraints)) + (10 .* .!vI .* rand(numConstraints)); 

        mP = blockdiag(spzeros(numElements, numElements), sparse(2I, numConstraints, numConstraints), spzeros(2 * numConstraints, 2 * numConstraints));
        vQ = [zeros(numElements + numConstraints); 2 .* ones(2 * numConstraints)];
        mIm = sparse(I, numConstraints, numConstraints);
        mA = [mAd -mIm -mIm mIm; spzeros(numConstraints, numElements + numConstraints) mIm spzeros(numConstraints, numConstraints); spzeros(numConstraints, numElements + numConstraints + numConstraints) mIm];
        vL = [vB; zeros(2 * numConstraints)];
        vU = [vB; Inf .* ones(2 * numConstraints)];
    elseif (problemClass == supportVectorMachine)
        densityFctr = 0.15;

        (numConstraints ≠ 0) || (numConstraints = numElements * 100);
        numClassA       = numConstraints ÷ 2;
        λ               = 1;
        vB = [ones(numClassA); -ones(numClassA)];
        mAu = sprandn(numClassA, numElements, densityFctr);
        mAl = sprandn(numClassA, numElements, densityFctr);
        mAd = [(mAu ./ sqrt(numConstraints)) + ((mAu .≠ 0)./ numConstraints); (mAl ./ sqrt(numConstraints)) - ((mAl .≠ 0) ./ numConstraints)];

        mP = blockdiag(sparse(2I, numElements, numElements), spzeros(numConstraints, numConstraints));
        vQ = λ .* [zeros(numElements); ones(numConstraints)];
        mA = [(spdiagm(0 => vB) * mAd) sparse(-I, numConstraints, numConstraints); spzeros(numConstraints, numElements) sparse(I, numConstraints, numConstraints)];
        vL = [-Inf .* ones(numConstraints); zeros(numConstraints)];
        vU = [-ones(numConstraints); Inf .* ones(numConstraints)];
    elseif (problemClass == isotonicRegression)
        densityFctr = 0.25;
        α           = 1e-2;

        mM = sprandn(numElements, numElements, densityFctr);
        mP = (mM' * mM) + sparse(α * I, numElements, numElements);
        vQ = randn(numElements);
        
        if (rand() >= 0.5)
            # Monotonic Non Increasing
            mA = spdiagm(numElements - 1, numElements, 0 => ones(numElements - 1), 1 => -ones(numElements - 1));
        else
            # Monotonic Non Decreasing
            mA = spdiagm(numElements - 1, numElements, 0 => -ones(numElements - 1), 1 => ones(numElements - 1));
        end
        vL = zeros(numElements - 1);
        vU = 10 * ones(numElements - 1);
    end

    return mP, vQ, mA, vL, vU;


end

