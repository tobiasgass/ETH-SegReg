function [x, fval, exitflag, output] = grad_solveTol(P, b, lambda, max_iters, optTol,progTol,method,x0)
% min for x:   1/2 |Px - b|_2^2 + lambda/2 |x|_2^2
% method(faster-->slower): 'csd', 'cg', 'pcg', 'lbfgs', .... 
%
% Example:
% addpath(genpath('minFunc_2012'));
% A = rand(1000, 100);
% x = rand(100, 1);
% b = A*x + randn(1000, 1) * 0.01;
% [xg, ~] = grad_solve(A, b, 0, 100, 'cg');
% xs = (A'*A) \ (A'*b);
% norm(xs - x)
% norm(xg - x)
    lambda=0.0;
    mfopts = []; 
    mfopts.method = method; % try to change for 'csd' it should be faster
    mfopts.display = 'iter';
    if max_iters>0
     mfopts.MaxIter = max_iters;
     mfopts.MaxFunEvals = 3 * max_iters;
    end
    %mfopts.LS_init = 2;
    if optTol>=0
      mfopts.optTol=optTol; %default 1e-6
    end
    if progTol>=0
      mfopts.progTol=progTol;%default 1e-9;
    end
    res=0.5*(norm(P*x0-b)^2);

    objf = @(ax) ls_f(ax, P, b, lambda,res);
 
    [x, fval,exitflag,output] = minFunc(objf, x0, mfopts);
       
end

function [f, g] = ls_f(x, P, b, lambda,res)
    Pxb = (P*x - b);
    f = sum(Pxb.^2)/2/res;
    g = (Pxb'*P)'/res; %faster than P'*Pxb since transpose of P is not needed
    if lambda ~= 0
        f = f + lambda/2 * sum(x.^2);
        g = g + lambda * x;
    end
end
