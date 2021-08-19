function [ vX, convFlag ] = SolveQuadraticProgram( vX, mP, vQ, mA, vL, vU, sSolverParams )
% ----------------------------------------------------------------------------------------------- %
%[ vX, convFlag ] = SolveQuadraticProgram( vX, mP, vQ, mA, vL, vU, sSolverParams )
% Solves: \arg \min_x 0.5 * x^T P x + x^T q 
%         subject to l <= A x <= u
% For a PSD Matrix P. It uses ADMM framework to solve the problem.
% Input:
%   - vX                -   Starting Point Vector.
%                           Used for initialization of the iterative
%                           solution.
%                           Structure: Vector (numElementsX x 1).
%                           Type: 'Double'.
%                           Range: (-inf, inf).
%   - mP                -   Model Matrix.
%                           The Quadratic Model Matrix. Must be a PSD
%                           Matrix.
%                           Structure: Matrix (numElementsX x numElementsX).
%                           Type: 'Double'.
%                           Range: (-inf, inf).
%   - vQ                -   Model Vector.
%                           The Quadratic Model vector.
%                           Structure: Vector (numElementsX x 1).
%                           Type: 'Double'.
%                           Range: (-inf, inf).
%   - mA                -   Constraints Matrix.
%                           The constraints matrix.
%                           Structure: Matrix (numConstraints x numElementsX).
%                           Type: 'Double'.
%                           Range: (-inf, inf).
%   - vL                -   Lower Boundary Vector.
%                           The lower boundary of the values of the linear
%                           constraints.
%                           Structure: Vector (numConstraints x 1).
%                           Type: 'Double'.
%                           Range: (-inf, inf).
%   - vU                -   Upper Boundary Vector.
%                           The upper boundary of the values of the linear
%                           constraints.
%                           Structure: Vector (numConstraints x 1).
%                           Type: 'Double'.
%                           Range: (-inf, inf).
%   - sSolverParams     -   Struct of Solver Parameters.
%                           Structure: Struct.
%                           Type: NA.
%                           Range: NA.
% Output:
%   - vX                -   Solution Vector.
%                           The solution of the solver (Given it worked).
%                           Structure: Vector (numElementsX x 1).
%                           Type: 'Double'.
%                           Range: (-inf, inf).
%   - convFlag          -   Convergence Flag.
%                           The status of convergence:
%                           1 - Exhausted iterations (No convergence).
%                           2 - Convergence of the vX (ADMM).
%                           3 - Primal and Dual Convergence.
%                           Type: 'Single' / 'Double'.
%                           Range {1, 2, 3}.
% References
%   1.  OSQP: An Operator Splitting Solver for Quadratic Programs (https://arxiv.org/abs/1711.08013).
% Remarks:
%   1.  The solver supports Sparse Matrices.
% Known Issues:
%   1.  C
% TODO:
%   1.  Implement scaling.
%   2.  Better reporting.
%   3.  Examine the use of Projected Gradient Descent for "Polishing". It
%       might be not feasible since it requires access to `mA` matrix row
%       by row (Slow for sparse matrices).
% Release Notes:
%   -   1.0.002     18/08/2021  Royi Avital
%       *   Fixed issue with `paramRhoInv` updating.
%       *   Added `EPS_ADMM_FACTOR` to prevent pre mature convergence.
%       *   Added clipping for `paramRho`.
%       *   Added logic for `LIN_SOLVER_MODE_AUTO`.
%   -   1.0.001     14/08/2021  Royi Avital
%       *   Added direct solver option.
%       *   Added adaptive update to Rho (As a scalar).
%   -   1.0.000     13/08/2021  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments
    vX (:, 1) {mustBeNumeric}
    mP (:, :) {mustBeNumeric}
    vQ (:, 1) {mustBeNumeric}
    mA (:, :) {mustBeNumeric}
    vL (:, 1) {mustBeNumeric}
    vU (:, 1) {mustBeNumeric}
    sSolverParams.numIterations (1, 1) {mustBeNumeric, mustBeReal, mustBePositive, mustBeInteger} = 4000
    sSolverParams.epsAbs (1, 1) {mustBeNumeric, mustBeReal, mustBePositive} = 1e-6
    sSolverParams.epsRel (1, 1) {mustBeNumeric, mustBeReal, mustBePositive} = 1e-6
    sSolverParams.paramRho (1, 1) {mustBeNumeric, mustBeReal, mustBePositive} = 1e6
    sSolverParams.paramSigma (1, 1) {mustBeNumeric, mustBeReal, mustBePositive} = 1e-6
    sSolverParams.paramAlpha (1, 1) {mustBeNumeric, mustBeReal, mustBePositive} = 1.6
    sSolverParams.paramDelta (1, 1) {mustBeNumeric, mustBeReal, mustBePositive} = 1e-6
    sSolverParams.adaptRho (1, 1) {mustBeNumeric, mustBeReal, mustBeInteger, mustBeMember(sSolverParams.adaptRho, [0, 1])} = 1
    sSolverParams.fctrRho (1, 1) {mustBeNumeric, mustBeReal, mustBePositive} = 5
    sSolverParams.numPolishItr (1, 1) {mustBeNumeric, mustBeReal, mustBeNonnegative, mustBeInteger} = 10
    sSolverParams.linSolverMode (1, 1) {mustBeNumeric, mustBeReal, mustBeInteger, mustBeMember(sSolverParams.linSolverMode, [1, 2, 3])} = 1
    sSolverParams.pcgEps (1, 1) {mustBeNumeric, mustBeReal, mustBePositive} = 1e-9 %<! Should be lower than min(epsAbs, epsRel)
    sSolverParams.pcgItr (1, 1) {mustBeNumeric, mustBeReal, mustBePositive} = 7500
    sSolverParams.minresEps (1, 1) {mustBeNumeric, mustBeReal, mustBePositive} = 1e-6
    sSolverParams.minresItr (1, 1) {mustBeNumeric, mustBeReal, mustBePositive} = 500
    sSolverParams.numItrConv (1, 1) {mustBeNumeric, mustBeReal, mustBeInteger, mustBePositive} = 50
end

FALSE   = 0;
TRUE    = 1;
OFF     = 0;
ON      = 1;

% Convergence status
CONV_FLAG_CONV_FAIL     = 1; %<! No convergence
CONV_FLAG_CONV_ADMM     = 2; %<! ADMM converged, problem not
CONV_FLAG_CONV_PRBLM    = 3; %<! Problem converged into feasible solution

% Linear system solver mode
LIN_SOLVER_MODE_AUTO        = 1; %<! Decide by the problem dimensions / number of non zeros
LIN_SOLVER_MODE_ITERATIVE   = 2; %<! Iterative solver
LIN_SOLVER_MODE_DIRECT      = 3; %<! Direct solver

EPS_ADMM_FACTOR = 1e-2;

% TODO: Numbers should be optimized with benchmark
MAX_NUM_ROWS_L  = 5000; %<! LKinear System dimensions
MAX_DENSITY     = 0.4;

MIN_VAL_RHO = 1e-3;
MAX_VAL_RHO = 1e6;

numIterations   = sSolverParams.numIterations; %<! Maximum number of iterations
epsAbs          = sSolverParams.epsAbs; %<! Absolute stopping threshold
epsRel          = sSolverParams.epsRel; %<! Relative stopping threshold
paramRho        = sSolverParams.paramRho; %<! Parameter Rho
paramSigma      = sSolverParams.paramSigma; %<! Parameter Sigma
paramAlpha      = sSolverParams.paramAlpha; %<! Parameter Alpha
paramDelta      = sSolverParams.paramDelta; %<! Parameter Delta
adaptRho        = sSolverParams.adaptRho; %<! Adaptive Step Size
fctrRho         = sSolverParams.adaptRho; %<! Adaptive update of Rho mode
numPolishItr    = sSolverParams.numPolishItr; %<! Number of polish iterations, if set to 0 no polish
linSolverMode   = sSolverParams.linSolverMode; %<! Linear system solver mode
pcgEps          = sSolverParams.pcgEps; %<! Threshold of the PCG solver (Iterative solver)
pcgItr          = sSolverParams.pcgItr; %<! Maximum iterations of iterative solver
minresEps       = sSolverParams.minresEps; %<! Threshold for the polish linear system solver (Iterative)
minresItr       = sSolverParams.minresItr; %<! Maximum iterations of the polish solver
numItrConv      = sSolverParams.numItrConv; %<! Number of iterations between convergence check up

% Check Dimesions
numElementsX            = size(vX, 1);
[numRowsP, numColsP]    = size(mP);
numElementsQ            = size(vQ, 1);
[numRowsA, numColsA]    = size(mA);
numElementsL            = size(vL, 1);
numElementsU            = size(vU, 1);

if(numElementsX ~= numColsP)
    error('The number of columns in matrix mP must match the number of elements of vX');
end

if(numRowsP ~= numColsP)
    error('The matrix mP must be a square matrix');
end

if(~issymmetric(mP))
    error('The matrix mP must be a symmetric positive definite matrix');
end

if(numElementsX ~= numElementsQ)
    error('The number of elements of vX must match the number of elements of vQ');
end

if(numElementsX ~= numColsA)
    error('The number of columns in matrix mA must match the number of elements of vX');
end

if(numElementsL ~= numRowsA)
    error('The number of elements of vL must match the number of rows of mA');
end

if(numElementsU ~= numRowsA)
    error('The number of elements of vU must match the number of rows of mA');
end

% Auxiliary Parameters
paramRhoInv = 1 / paramRho;
paramAlpha1 = 1 - paramAlpha;

switch(linSolverMode)
    case(LIN_SOLVER_MODE_AUTO)
        numRowsL    = numRowsP + numRowsA;
        numNonZeros = nnz(mP) + nnz(mA);
        nnzDensity  = numNonZeros / (numRowsL * numRowsL);
        if((numRowsL <= MAX_NUM_ROWS_L) && (nnzDensity <= MAX_DENSITY))
            directSol = true;
        else
            directSol = false;
        end
    case(LIN_SOLVER_MODE_ITERATIVE)
        directSol = OFF;
    case(LIN_SOLVER_MODE_DIRECT)
        directSol = ON;
end

convFlag = CONV_FLAG_CONV_FAIL;

epsAdmm = min(epsAbs, epsRel) * EPS_ADMM_FACTOR;

vXX = vX;
vXP = zeros(numElementsX, 1); %<! Previous iteration of vX
vZ  = zeros(numRowsA, 1);
vY  = zeros(numRowsA, 1);
vZZ = zeros(numRowsA, 1);
vZP = zeros(numRowsA, 1); %<! Previous iteration of vZ

if(adaptRho == ON)
    paramRhoA = paramRho;
    
    mAA = mA.' * mA;
    mPI = mP + (paramSigma * speye(numElementsX));    
end

if(directSol == ON)
    hDL = decomposition([mP + (paramSigma * speye(numElementsX)), mA.'; mA, -paramRhoInv * speye(numRowsA)], 'ldl');
else
    % TODO: Search if there are better options for this structure of a matrix
    % than Preconditioned Conjugate Gradient.
    % mL is Positive Definite Matrix
    mL = mP + (paramSigma * speye(numElementsX)) + (paramRho * (mA.' * mA));
end

for ii = 1:numIterations
     if((adaptRho == ON) && ((paramRhoA * fctrRho < paramRho) || (paramRhoA > fctrRho * paramRho)))
        paramRho    = paramRhoA;
        paramRhoInv = 1 / paramRho;
        if(directSol == ON)
            hDL = decomposition([mP + (paramSigma * speye(numElementsX)), mA.'; mA, -paramRhoInv * speye(numRowsA)], 'ldl');
        else
            mL = mPI + (paramRho * mAA);
        end
     end
    
    if(directSol == ON)
        vV = hDL \ [paramSigma * vX - vQ; vZ - paramRhoInv * vY];
        vXX(:) = vV(1:numElementsX);
        vZZ(:) = vZ + paramRhoInv * (vV((numElementsX + 1):end) - vY);
    else
        [vXX(:), pcgFlag] = pcg(mL, paramSigma * vX - vQ + mA.' * (paramRho * vZ - vY), pcgEps, pcgItr, [], [], vX);
        vZZ(:) = mA * vXX;
    end
   
    vXP(:) = vX;
    vX(:)  = (paramAlpha * vXX) + (paramAlpha1 * vX);
    vZP(:) = vZ; %<! Keeping vZ of previous iteration
    vZ(:)  = min(max(paramAlpha * vZZ + paramAlpha1 * vZ + paramRhoInv * vY, vL), vU);
    vY(:)  = vY + paramRho * (paramAlpha * vZZ + paramAlpha1 * vZP - vZ);
    
    if(mod(ii, numItrConv) == 0)
        % Pre Computation
        normResPrim = norm(mA * vX - vZ, 'inf');
        normResDual = norm(mP * vX + vQ + mA.' * vY, 'inf');
        
        maxNormPrim = max([norm(mA * vX, 'inf'), norm(vZ, 'inf')]);
        maxNormDual = max([norm(mP * vX, 'inf'), norm(mA.' * vY, 'inf'), norm(vQ, 'inf')]);
        
        % Adaptive Rho
        if(adaptRho == ON)
            numeratorVal    = normResPrim * maxNormDual;
            denominatorVal  = normResDual * maxNormPrim;
            paramRhoA       = min(max(paramRho * sqrt(numeratorVal / denominatorVal), MIN_VAL_RHO), MAX_VAL_RHO);
        end
        % Termination
        epsPrim = epsAbs + epsRel * maxNormPrim;
        epsDual = epsAbs + epsRel * maxNormDual;
        
        if((normResPrim < epsPrim) && (normResDual < epsDual))
            convFlag = CONV_FLAG_CONV_PRBLM;
            break;
        end
        if((norm(vX - vXP, 'inf') <= epsAdmm) && (norm(vZ - vZP, 'inf') <= epsAdmm))
            convFlag = CONV_FLAG_CONV_ADMM;
            break;
        end
    end
    
end

% Polishing
% Works if the active constraints (By the lagrange multiplier - `vY`) are
% identified correctly.
if(numPolishItr > 0)
    vLi = vY < 0;
    vUi = vY > 0;
    
    numL = sum(vLi);
    numU = sum(vUi);
    
    vG = [-vQ; vL(vLi); vU(vUi)];
    
    mAL = mA(vLi, :);
    mAU = mA(vUi, :);
    
    mK  = [mP, mAL.', mAU.'; mAL, sparse(numL, numL), sparse(numL, numU); mAU, sparse(numU, numL), sparse(numU, numU)];
    mKK = mK + blkdiag(paramDelta * speye(numElementsX), -paramDelta * speye(numL), -paramDelta * speye(numU));
    
    vT  = zeros(numElementsX + numL + numU, 1);
    vTT = zeros(numElementsX + numL + numU, 1);
end

minresFlag = -1;

% Polishing
for jj = 1:numPolishItr
    [vTT(:), minresFlag] = minres(mKK, vG - mK * vT, minresEps, minresItr, [], [], vTT);
    if(minresFlag)
        break;
    end
    vT = vT + vTT;
end

if(~minresFlag)
    % Update solution only if the linear system solution converged.
    vX = vT(1:numElementsX);
end


end

