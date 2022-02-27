[![Visitors](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FRoyiAvital%2FStackExchangeCodes&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitors+%28Daily+%2F+Total%29&edge_flat=false)](https://github.com/RoyiAvital/QuadraticProgramSolver)
<a href="https://liberapay.com/Royi/donate"><img alt="Donate using Liberapay" src="https://liberapay.com/assets/widgets/donate.svg"></a>

# Quadratic Program Solver
[![View Quadratic Program Solver on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange/97899)

Solves a [Quadratic Programming][001] problem using [Alternating Direction Method of Multipliers](https://en.wikipedia.org/wiki/Augmented_Lagrangian_method) (ADMM).  
This is a MATLAB implementation of the paper - [OSQP: An Operator Splitting Solver for Quadratic Programs][002].

**Remark**: Any Quadratic Program Solver can solve [Constrained Least Squares](https://en.wikipedia.org/wiki/Constrained_least_squares) problem as well (With linear and convex constraints).

## Motivation

I needed for some Signal / Image Processing projects a solver of a problem of the form:

$$\begin{aligned}
\arg \min_{\boldsymbol{x}} & \quad \frac{1}{2} {\left\| A \boldsymbol{x} - \boldsymbol{b} \right\|}_{2}^{2} \\
\text{subject to} & \quad B \boldsymbol{x} \leq \boldsymbol{c} \\
& \quad D \boldsymbol{x} = \boldsymbol{e}
\end{aligned}$$

I could use MATLAB's [`quadprog()`](https://www.mathworks.com/help/optim/ug/quadprog.html) or [`lsqlin()`](https://www.mathworks.com/help/optim/ug/lsqlin.html) yet both are part of the [Optimization Toolbox][003] which isn't widely accessible.  
When I learned about ADMM, Projection and Optimization in general I played with some implementations for this problem but they were pretty slow and sensitive to parameters.  
When [OSQP](https://osqp.org/) became available it showed those problem can be solved.  
It requires compiling and even defining `GCC` as the compiler on Windows which isn't easy to everyone.  
Hence I thought it would be nice to replicate the paper (As the `C` code is way beyond me) in MATLAB.  
Within few hours I had a first working code though without all the features (See [To Do](#to-do)).  

The goal is to have a viable alternative to MATLAB's `quadprog()`.  
While in most cases `quadprog()` will be faster choice, this implementation should be _good enough_ and free solution.

Implementation in _High Level Language_ might assist with faster integration of better optimization and flexibility (For instance, supporting non sparse matrices).

## The Code  

The solver is implemented in the function `SolveQuadraticProgram()`.  
It uses MATLAB's `argument` block and requires `MATLAB R2020b (Ver 9.9)` to the least.  
Users of previous MATLAB versions might try remove this block (Be careful about the parameters).

The function solves the following form of [Quadratic Program][001]:

$$\begin{aligned}
\arg \min_{\boldsymbol{x}} & \quad \frac{1}{2} \boldsymbol{x}^{T} P \boldsymbol{x} + \boldsymbol{x}^{T} \boldsymbol{q} \\
\text{subject to} & \quad \boldsymbol{l} \leq A \boldsymbol{x} \leq \boldsymbol{u}
\end{aligned}$$

Where $ P \in \mathbb{S}_{+}^{n} $ (A symmetric positive semi definite matrix).  

## Documentation

> In Progress...

Look inside the function `SolveQuadProgram()` and see the [reference paper][002].

## Unit Test

The unit tests implemented in `SolveQuadraticProgramUnitTest.m`.  
The script requires [CVX](http://cvxr.com/) and [Optimization Toolbox][003].  
It basically generates a problem and verify the solution of `SolveQuadraticProgram()` vs. the other 2 references.

## To Do

1.	Check if making `paramRho` a matrix has a real benefit.
2.	Implement the scaling procedure from the reference paper.
3.	Optimize the solution to the linear system (Is there anything better than `pcg()` for this structure? [LSMR](https://web.stanford.edu/group/SOL/software/lsmr/) style?).
4.	Better reporting.
5.	Implement last step as Projected Gradient Descent in order to be strictly feasible.

## Julia Code

The project implements the method in [Julia Language](https://julialang.org/) as well (The `.jl` files).  
The goal is to have a Julia package on its own. But it will happen only once performance will be in parity with the `C` code.  
Julia has the potential to have even better performance than MATLAB's `quadprog()`.

## Reference

The code is an implementation of the paper [Stellato, B. and Banjac, G. and Goulart, P. and Bemporad, A. and Boyd, S. - {OSQP}: An Operator Splitting Solver for Quadratic Programs][002].  
This is a really great paper as the writers gave all the little details to create a truly competitive solver with ADMM.  
Their work is really amazing.


  [001]: https://en.wikipedia.org/wiki/Quadratic_programming
  [002]: https://arxiv.org/abs/1711.08013
  [003]: https://www.mathworks.com/products/optimization.html
