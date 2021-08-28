using JuMP;
using Gurobi;

function SolveQpJump(mP, vQ, mA, vL, vU; hOptFun = nothing)
    
    if (isnothing(hOptFun))
        hOptFun = Gurobi.Optimizer;
    end
   
    jumpModel = Model(hOptFun);
    @variable(jumpModel, vT[1:numElements]);
    @objective(jumpModel, Min, 0.5 * (vT' * mP * vT) + (vT' * vQ));
    vV = vL .≠ -Inf;
    @constraint(jumpModel, vL[vV] .<= mA[vV, :] * vT);
    vV .= vU .≠ Inf;
    @constraint(jumpModel, mA[vV, :] * vT .<= vU[vV]);
    optimize!(jumpModel);
    
    return value.(vT);
    
end