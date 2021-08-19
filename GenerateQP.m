function [ mP, vQ, mA, vL, vU ] = GenerateQP( problemClass, numElements, numConstraints )
% ----------------------------------------------------------------------------------------------- %
%[ mP, vQ, mA, vL, vU ] = GenerateQP( problemClass, numElements, numConstraints )
% Generates Quadratic Program of the form: 
%       \arg \min_x 0.5 * x^T P x + x^T q 
%       subject to l <= A x <= u
% For a PSD Matrix P.
% Input:
%   - problemClass  -   Problem Class / Type.
%                       The problems type before it is converted into
%                       Quadratic Program form.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, 3, ..., 9}.
%   - numElements   -   Number of Elements.
%                       The number of elements in vector `vX`. In some
%                       problems it will be used bu the final number of
%                       rows of `mP` might be different.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, 3, ...}.
%   - numElements   -   Number of Constraints.
%                       The number of elements in vector `vU` / `vL`. In
%                       some problems it will be used but the final number
%                       of rows in `mA` might be different.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, 3, ...}.
% Output:
%   - mP            -   Model Matrix.
%                       The Quadratic Model Matrix. Must be a PSD Matrix.
%                       Structure: Matrix (numElementsX x numElementsX).
%                       Type: 'Double'.
%                       Range: (-inf, inf).
%   - vQ            -   Model Vector.
%                       The Quadratic Model vector.
%                       Structure: Vector (numElementsX x 1).
%                       Type: 'Double'.
%                       Range: (-inf, inf).
%   - mA            -   Constraints Matrix.
%                       The constraints matrix.
%                       Structure: Matrix (numConstraints x numElementsX).
%                       Type: 'Double'.
%                       Range: (-inf, inf).
%   - vL            -   Lower Boundary Vector.
%                       The lower boundary of the values of the linear
%                       constraints.
%                       Structure: Vector (numConstraints x 1).
%                       Type: 'Double'.
%                       Range: (-inf, inf).
%   - vU            -   Upper Boundary Vector.
%                       The upper boundary of the values of the linear
%                       constraints.
%                       Structure: Vector (numConstraints x 1).
%                       Type: 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  OSQP: An Operator Splitting Solver for Quadratic Programs (https://arxiv.org/abs/1711.08013).
% Remarks:
%   1.  All output matrices are sparse.
%   2.  The actual number of rows of `mP` / `mA` might be different from
%       the input due to the conversion from the original form to QP form.
% Known Issues:
%   1.  C
% TODO:
%   1.  Implement the missing methods.
% Release Notes:
%   -   1.0.000     19/08/2021  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments
    problemClass (1, 1) {mustBeNumeric, mustBeReal, mustBePositive, mustBeMember(problemClass, [1:9])} = 1
    numElements (1, 1) {mustBeNumeric, mustBeReal, mustBePositive, mustBeInteger} = 100
    numConstraints (1, 1) {mustBeNumeric, mustBeReal, mustBePositive, mustBeInteger} = 50
end

FALSE   = 0;
TRUE    = 1;
OFF     = 0;
ON      = 1;

PROBLEM_CLASS_RADNOM_QP                 = 1;
PROBLEM_CLASS_EQUALITY_CONSTRAINED_QP   = 2;
PROBLEM_CLASS_OPTIMAL_CONTROL           = 3;
PROBLEM_CLASS_PORTFOLIO_OPTIMIZATION    = 4;
PROBLEM_CLASS_LASSO_OPTIMIZATION        = 5;
PROBLEM_CLASS_HUBBER_FITTING            = 6;
PROBLEM_CLASS_SUPPORT_VECTOR_MACHINE    = 7;
PROBLEM_CLASS_RANDOM_QP_WITH_EQL_CONS   = 8;
PROBLEM_CLASS_ISOTONIC_REGRESSION       = 9;

switch(problemClass)
    case({PROBLEM_CLASS_RADNOM_QP, PROBLEM_CLASS_EQUALITY_CONSTRAINED_QP, PROBLEM_CLASS_RANDOM_QP_WITH_EQL_CONS})
        densityFctr = 0.15;
        paramAlpha  = 1e-2;
        
        mM = sprandn(numElements, numElements, densityFctr);
        mP = mM.' * mM + paramAlpha * speye(numElements, numElements);
        vQ = randn(numElements, 1);
        
        mA = sprandn(numConstraints, numElements, densityFctr);
        if(problemClass == PROBLEM_CLASS_RADNOM_QP)
            vL = -rand(numConstraints, 1);
            vU = rand(numConstraints, 1);
        elseif(problemClass == PROBLEM_CLASS_EQUALITY_CONSTRAINED_QP)
            vL = (2 * rand(numConstraints, 1)) - 1;
            vU = vL;
        else
            vL = -rand(numConstraints, 1);
            vU = rand(numConstraints, 1);
            vI = rand(numConstraints, 1) <= 0.15;
            vL(vI) = vU(vI);
            vI = rand(numConstraints, 1) <= 0.15;
            vU(vI) = vI(vI);
        end
    case(PROBLEM_CLASS_PORTFOLIO_OPTIMIZATION)
        mD = spdiags(sqrt(numConstraints) * rand(numElements, 1), 0, numElements, numElements);
        mP = [mD, sparse(numElements, numConstraints); sparse(numConstraints, numElements), speye(numConstraints, numConstraints)];
        vQ = [randn(numElements, 1); zeros(numConstraints, 1)];
        
        mF = sprandn(numElements, numConstraints, 0.5);
        mA = [mF.' -speye(numConstraints, numConstraints); ones(1, numElements) sparse(1, numConstraints); speye(numElements, numElements) sparse(numElements, numConstraints)];
        vL = [zeros(numConstraints, 1); 1; zeros(numElements, 1)];
        vU = [zeros(numConstraints, 1); 1; ones(numElements, 1)];
    case(PROBLEM_CLASS_ISOTONIC_REGRESSION)
        densityFctr = 0.25;
        paramAlpha  = 1e-2;
        
        mM = sprandn(numElements, numElements, densityFctr);
        mP = mM.' * mM + paramAlpha * speye(numElements, numElements);
        vQ = randn(numElements, 1);
        
        if(rand(1, 1) >= 0.5)
            % Monotonic Non Increasing
            mA = spdiags([ones(numElements, 1), -ones(numElements, 1)], [0, 1], numElements - 1, numElements);
        else
            % Monotonic Non Decreasing
            mA = spdiags([-ones(numElements, 1), ones(numElements, 1)], [0, 1], numElements - 1, numElements);
        end
        vL = zeros(numElements - 1, 1);
        vU = 10 * ones(numElements - 1, 1);
end


end

