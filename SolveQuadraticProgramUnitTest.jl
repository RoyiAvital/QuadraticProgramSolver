# Unit Test for `SolveQuadraticProgram()`

using LinearAlgebra
using SparseArrays
using Convex
using SCS
# using ECOS #<! Seems to be very inaccurate for this problem
# using Gurobi
using Plots
using Random
using MAT

# For manual Gurobi installation
# ENV["GUROBI_HOME"] = "D:\\Applications\\Gurobi"
# ENV["GRB_LICENSE_FILE"] = "D:\\Applications\\Gurobi\\gurobi.lic"

# Random.seed!(1234);

include("GenerateQuadraticProgram.jl");
include("SolveQuadraticProgram.jl");

@enum ProblemClass randomQp equalityConstrainedQp optimalControl portfolioOptimization lassoOptimization huberFitting supportVectorMachine
@enum DataSource dataSourceGenerated = 1 dataSourceLoaded

## Parameters

# Simulaion
numSimulations  = 10;
numElements     = 100;
numConstraints  = 50;

dataSource      = dataSourceGenerated;
dataFileName    = "QpModel.mat";

problemClass    = rand(instances(ProblemClass));
# problemClass    = equalityConstrainedQp;

# Solver
numIterations   = 450;
ρ               = 1000000.01;
adptΡ           = true;
linSolverMode   = modeDirect;

if (dataSource == dataSourceGenerated)
    mP, vQ, mA, vL, vU = GenerateRandomQP(problemClass, numElements, numConstraints);
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
solve!(hPrblm, SCS.Optimizer(eps = 1e-8); silent_solver = true);

# vX = copy(vT.value);
convFlag = SolveQuadraticProgram!(vX, mP, vQ, mA, vL, vU; numIterations = numIterations, ρ = ρ, adptΡ = adptΡ, linSolverMode = linSolverMode);

maxAbsDev = norm(vT.value - vX, Inf);

display(scatter([vX, vT.value], title = "Max Absolute Deviation: $(maxAbsDev)", label = ["Solver" "Reference"]));
display(scatter([mA * vT.value, vL, vU], label = ["Solver" "Lower Boundary" "Upper Boundary"]));

println(maxAbsDev)