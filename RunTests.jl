# Unit Test for `SolveQuadraticProgram()`

using LinearAlgebra;
using SparseArrays;
using Gurobi; #<! Solves problems which SCS struggles with
using OSQP;
using Random;
using StableRNGs;
using Test;

seedNumber = 1234;
Random.seed!(seedNumber);
# Random.seed!(StableRNG(seedNumber), seedNumber);

include("GenerateQuadraticProgram.jl");
include("SolveQuadraticProgram.jl");
include("LinearSystemSolvers.jl");
include("SolveQuadraticProgramJump.jl");

# @enum ProblemClass randomQp = 1 inequalityConstrainedQp equalityConstrainedQp optimalControl portfolioOptimization lassoOptimization huberFitting supportVectorMachine isotonicRegression
@enum RefSolver gurobiSolver = 1 osqpSolver

## Parameters

# Simulaion

refSolver = osqpSolver;

numSimulations  = 10;
mNumElements = [    0010 0100;
0010 0100;
0010 0100;
0010 0100;
0010 0100;
0010 0100;
0010 0100;
0010 0100;
0010 0100;];
mNumConstraints = [ 0000 0000;
0000 0000;
0005 0050;
0000 0000
0000 0000
0000 0000
0000 0000
0000 0000
0000 0000]; #<! Set to 0 for OSQP Paper dimensions

# Solver
numIterations   = 50000;
ϵAbs            = 1e-7;
ϵRel            = 1e-7;
ρ               = 0.1;
adptΡ           = true;
hLinSolInit     = FacLdlInit;
hLinSol         = FacLdl!;

absDevThr = 1e-5;

hOptFun = optimizer_with_attributes(Gurobi.Optimizer, "OptimalityTol" => 1e-8, "FeasibilityTol" => 1e-8);

@testset verbose = true "Unit Test vs. Gurobi" begin
for problemClass ∈ instances(ProblemClass)
    @testset "Unit Test of $(problemClass) Problem Class" begin
    for iSim ∈ 1:numSimulations
        for iDim ∈ 1:size(mNumElements, 2)
            mP, vQ, mA, vL, vU = GenerateRandomQP(problemClass, mNumElements[Int(problemClass), iDim]; numConstraints = mNumConstraints[Int(problemClass), iDim]);
            # Some problems might change the actual data
            numElements     = size(mP, 1);
            numConstraints  = size(mA, 1);
            
            vX = zeros(numElements);
            
            if (refSolver == gurobiSolver)
                runTimeRefSolver = @elapsed vT = SolveQpJump(mP, vQ, mA, vL, vU; hOptFun = hOptFun);
            elseif (refSolver == osqpSolver)
                runTimeRefSolver = @elapsed begin
                    osqPModel = OSQP.Model();
                    OSQP.setup!(osqPModel; P = mP, q = vQ, A = mA, l = vL, u = vU, rho = ρ, max_iter = numIterations, eps_abs = 1e-7, eps_rel = 1e-7, scaling = 0);
                    osqpRes = OSQP.solve!(osqPModel);
                    vT = osqpRes.x;
                end
            end

            runTime     = @elapsed convFlag = SolveQuadraticProgram!(vX, mP, vQ, mA, vL, vU, hLinSolInit, hLinSol; numIterations = numIterations, ϵAbs = ϵAbs, ϵRel = ϵRel, ρ = ρ, adptΡ = adptΡ);

            maxAbsDev = norm(vT - vX, Inf);
            
            println("The max absolute error is: $(maxAbsDev)");
            println("The run time is: $(runTime) [Sec]");
            println("The output flag is: $(convFlag)");
            println("The reference solver ($(refSolver)) run time is: $(runTimeRefSolver) [Sec]");
            @test maxAbsDev <= absDevThr;
        end
    end
end #<! End Test Set Problem Class
end

end #<! Test Set Unit Test