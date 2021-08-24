# Unit Test for `SolveQuadraticProgram()`

using LinearAlgebra
using SparseArrays
using Convex
using SCS
# using ECOS #<! Seems to be very inaccurate for this problem
using Gurobi
using Plots
using Random
using StableRNGs
using MAT

# For manual Gurobi installation
ENV["GUROBI_HOME"] = "D:\\Applications\\Gurobi"
ENV["GRB_LICENSE_FILE"] = "D:\\Applications\\Gurobi\\gurobi.lic"


seedNumber = 1234;
Random.seed!(seedNumber);
# Random.seed!(StableRNG(seedNumber), seedNumber);

include("GenerateQuadraticProgram.jl");
include("SolveQuadraticProgram.jl");

# @enum ProblemClass randomQp = 1 inequalityConstrainedQp equalityConstrainedQp optimalControl portfolioOptimization lassoOptimization huberFitting supportVectorMachine isotonicRegression
@enum DataSource dataSourceGenerated = 1 dataSourceLoaded

## Parameters

# Simulaion
numSimulations  = 10;
numElements     = 50;
numConstraints  = 0; #<! Set to 0 for OSQP Paper dimensions

dataSource      = dataSourceGenerated;
dataFileName    = "QpModel.mat";

# problemClass    = rand(instances(ProblemClass));
problemClass    = isotonicRegression;

# Solver
numIterations   = 5000;
ρ               = 1e6
adptΡ           = true;
linSolverMode   = modeDirect;

if (dataSource == dataSourceGenerated)
    mP, vQ, mA, vL, vU = GenerateRandomQP(problemClass, numElements; numConstraints = numConstraints);
else
    dMatFile = matread(dataFileName); 
    mP = dMatFile["mP"];
    vQ = dropdims(dMatFile["vQ"]; dims = 2);
    mA = dMatFile["mA"];
    vL = dropdims(dMatFile["vL"]; dims = 2);
    vU = dropdims(dMatFile["vU"]; dims = 2);
end

# Some problems might change the actual data
numElements     = size(mP, 1);
numConstraints  = size(mA, 1);

vX = zeros(numElements);

fObjFun(vX) = 0.5 * dot(vX, mP, vX) + dot(vQ, vX);

# Reference: Convex Solver
vT = Variable(numElements);
hPrblm = minimize(0.5 * quadform(vT, Matrix(mP)) + dot(vT, vQ), [vL <= Matrix(mA) * vT, Matrix(mA) * vT <= vU]);
# runTimeCvx = @elapsed Convex.solve!(hPrblm, SCS.Optimizer(eps = 1e-8); silent_solver = true);
# Gurobi Parameters: https://www.gurobi.com/documentation/9.1/refman/parameters.html
runTimeCvx = @elapsed Convex.solve!(hPrblm, Gurobi.Optimizer(OptimalityTol = 1e-8, FeasibilityTol = 1e-8); silent_solver = false);

# vX = copy(vT.value);
runTime = @elapsed convFlag = SolveQuadraticProgram!(vX, mP, vQ, mA, vL, vU; numIterations = numIterations, ρ = ρ, adptΡ = adptΡ, linSolverMode = linSolverMode);

maxAbsDev = norm(vT.value - vX, Inf);

display(scatter([vX, vT.value], title = "Solver Solution\nMax Absolute Deviation: $(maxAbsDev)\nRun Time: $(runTime) [Sec], Solver Mode: $(linSolverMode)", label = ["Solver" "Reference"]));
display(scatter([mA * vT.value, vL, vU], title = "Constraints Map", label = ["Solver" "Lower Boundary" "Upper Boundary"]));

println("The max absolute error is: $(maxAbsDev)");
println("The run time is: $(runTime) [Sec]");
println("The output flag is: $(convFlag)");
println("The Convex.jl run time is: $(runTimeCvx) [Sec]");