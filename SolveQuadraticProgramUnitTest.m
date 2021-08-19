% Solve Quadratic Program Unit Test
% A unit test for `SolveQuadraticProgram()` using a generated cases as in
% the paper OSQP: An Operator Splitting Solver for Quadratic Programs.
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  Integrate OSQP test suite (https://github.com/osqp/osqp_benchmarks).
% Release Notes
% - 1.0.000     15/08/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 5191;79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

DATA_SOURCE_GENERATED   = 1;
DATA_SOURCE_LOADED      = 2;

PROBLEM_CLASS_RADNOM_QP                 = 1;
PROBLEM_CLASS_EQUALITY_CONSTRAINED_QP   = 2;
PROBLEM_CLASS_OPTIMAL_CONTROL           = 3;
PROBLEM_CLASS_PORTFOLIO_OPTIMIZATION    = 4;
PROBLEM_CLASS_LASSO_OPTIMIZATION        = 5;
PROBLEM_CLASS_HUBBER_FITTING            = 6;
PROBLEM_CLASS_SUPPORT_VECTOR_MACHINE    = 7;
PROBLEM_CLASS_RANDOM_QP_WITH_EQL_CONS   = 8; %<! Both equality and inequality
PROBLEM_CLASS_ISOTONIC_REGRESSION       = 9; %<! Ignores `numConstraints`

LIN_SOLVER_MODE_AUTO        = 1; %<! Decide by the problem dimensions / number of non zeros
LIN_SOLVER_MODE_ITERATIVE   = 2; %<! Iterative solver
LIN_SOLVER_MODE_DIRECT      = 3; %<! Direct solver


%% Simulation Parameters

% Data Parameters
dataSource      = DATA_SOURCE_GENERATED;
dataFileName    = 'QpModel.mat';
exportModel     = OFF;

% Problem Generation
problemClass    = PROBLEM_CLASS_ISOTONIC_REGRESSION;
numSimulations  = 10;
numElements     = 1000;
numConstraints  = 500;

% Solver Parameters
numIterations   = 5000;
epsVal          = 1e-9;
paramRho        = 1e6;
adaptRho        = ON;
numPolishItr    = 10;
linSolverMode   = LIN_SOLVER_MODE_ITERATIVE;


%% Generate Data

switch(dataSource)
    case(DATA_SOURCE_GENERATED)
        [mP, vQ, mA, vL, vU]    = GenerateQP(problemClass, numElements, numConstraints);
    case(DATA_SOURCE_LOADED)
        load(dataFileName);
        exportModel = OFF;
    otherwise
        error('Invalid value of `dataSource`');
end

numElements     = size(mP, 1);
numConstraints  = size(mA, 1);

vX = zeros(numElements, 1);

if(exportModel == ON)
    save('QpModel', 'mP', 'vQ', 'mA', 'vL', 'vU');
end

hObjFun = @(vX) (0.5 * (vX.' * mP * vX)) + (vQ.' * vX);


%% Analysis

hRunTime = tic();
[vXX, convFlag] = SolveQuadraticProgram(vX, mP, vQ, mA, vL, vU, ...
    'numIterations', numIterations, 'epsRel', epsVal, 'epsAbs', epsVal, ...
    'paramRho', paramRho, 'adaptRho', adaptRho, 'numPolishItr', numPolishItr, ...
    'linSolverMode', linSolverMode);
runTime = toc(hRunTime);

vSol = vXX;
disp(['SolveQuadraticProgram() Analysis']);
disp(['Run Time: ', num2str(runTime), ' [Sec]']);
disp(['Obj Val: ', num2str(hObjFun(vSol)), ', L Violation: ', num2str(min(mA * vSol - vL)), ', U Violation: ', num2str(max(mA * vSol - vU))]);
disp(['']);


sOpt    = optimoptions('quadprog', 'Display', 'off');
hRunTime = tic();
vYY     = quadprog(mP, vQ, [-mA; mA], [-vL; vU], [], [], [], [], vX, sOpt);
runTime = toc(hRunTime);

vSol = vYY;
disp(['quadprog() Analysis']);
disp(['Run Time: ', num2str(runTime), ' [Sec]']);
disp(['Obj Val: ', num2str(hObjFun(vSol)), ', L Violation: ', num2str(min(mA * vSol - vL)), ', U Violation: ', num2str(max(mA * vSol - vU))]);
disp(['']);

cvx_solver('SDPT3'); %<! Default, Keep numRows low
% cvx_solver('SeDuMi');
% cvx_solver('Mosek'); %<! Can handle numRows > 500, Very Good!
% cvx_solver('Gurobi');
hRunTime = tic();
cvx_begin('quiet')
    cvx_precision('best');
    variable vZZ(numElements, 1);
    minimize( (0.5 * quad_form(vZZ, mP)) + (vZZ.' * vQ) );
    subject to
        vL <= mA * vZZ <= vU;
cvx_end

runTime = toc(hRunTime);

vSol = vZZ;
disp(['CVX) Analysis']);
disp(['Run Time: ', num2str(runTime), ' [Sec]']);
disp(['Obj Val: ', num2str(hObjFun(vSol)), ', L Violation: ', num2str(min(mA * vSol - vL)), ', U Violation: ', num2str(max(mA * vSol - vU))]);
disp(['']);


%% Display Results

figure();
scatter(1:numElements, [vXX, vYY, vZZ]);
hLegend = ClickableLegend({['Solver'], ['quadprog()'], ['CVX']});
norm(vXX - vYY, 'inf')
norm(vXX - vZZ, 'inf')

% figureIdx = figureIdx + 1;
% 
% hFigure = figure('Position', figPosLarge);
% hAxes   = axes(hFigure);
% hLineObj = plot(vSnrdB, 10 * log10([vFreqMseCrlb, mFreqErr]));
% set(hLineObj, 'LineWidth', lineWidthNormal);
% % set(hLineObj(1), 'LineStyle', 'none', 'Marker', '*');
% % set(hLineObj(2), 'LineStyle', 'none', 'Marker', 'x');
% set(hAxes, 'YLim', [-120, 0]);
% set(get(hAxes, 'Title'), 'String', {['MSE of Sine Frequency Estimation'], ...
%     ['Number of Samples: ', num2str(numSamples), ', Relative Frequncy [Fc / Fs]: ', num2str(sineFreq / samplingFreq), ...
%     ', Number of Realizations: ', num2str(numRealizations)]}, ...
%     'FontSize', fontSizeTitle);
% set(get(hAxes, 'XLabel'), 'String', {['SNR [dB]']}, ...
%     'FontSize', fontSizeAxis);
% set(get(hAxes, 'YLabel'), 'String', {['MSE']}, ...
%     'FontSize', fontSizeAxis);
% hLegend = ClickableLegend({['CRLB'], ['Kay Estimator Type 1'], ['Kay Estimator Type 2']});
% 
% if(generateFigures == ON)
%     % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
%     print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
% end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

