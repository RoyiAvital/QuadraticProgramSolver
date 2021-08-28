using JuMP;
using Gurobi;

function SolveQpJump(mP, vQ, mA, vL, vU; hOptFun = nothing, verbosFlag = false)
    #TODO: See https://discourse.julialang.org/t/67214/5
    # Check if it gives speed improvements.
    
    if (isnothing(hOptFun))
        hOptFun = Gurobi.Optimizer;
    end
   
    jumpModel = Model(hOptFun);
    if (verbosFlag)
        unset_silent(jumpModel);
    else
        set_silent(jumpModel);
    end
    @variable(jumpModel, vT[1:numElements]);
    @objective(jumpModel, Min, 0.5 * (vT' * mP * vT) + (vT' * vQ));
    vV = vL .≠ -Inf;
    @constraint(jumpModel, vL[vV] .<= mA[vV, :] * vT);
    vV .= vU .≠ Inf;
    @constraint(jumpModel, mA[vV, :] * vT .<= vU[vV]);
    optimize!(jumpModel);
    
    return value.(vT);
    
end