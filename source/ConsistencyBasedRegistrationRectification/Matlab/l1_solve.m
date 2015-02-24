function [x, fval] = l1_solve(P, b, lambda, max_iters,optTol,progTol, x0)
%wrapper for minFunc
    mfopts = [];
    mfopts.display = 'iter';
    if max_iters>0
     mfopts.MaxIter = max_iters;
     mfopts.MaxFunEvals = 3 * max_iters;
    end
     if optTol>=0   
      mfopts.optTol=optTol; %default 1e-6
    end
    if progTol>=0
      mfopts.progTol=progTol;%default 1e-9;
    end
   %objective function pointer
    objf = @(ax) ls_fTV(ax, P, b);%, lambda);
   %call to minFunc
    [x] = L1General2_PSSgb(objf,x0,lambda,mfopts);
end


function [f, g] = ls_fTV(x, P, b)%, lambda)
    Pxb = P*x - b;
    f = sum(Pxb.^2)/2;
    g = (Pxb'*P)'; %more efficient than P'*Pxb since transposing P is avoided
end