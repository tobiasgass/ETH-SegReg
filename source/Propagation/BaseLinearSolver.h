#pragma once
#include "engine.h"
#include "matrix.h"


//pure virtual class
//solves Ax=b
//creation of A,x,b needs to be handled by derivatives of this class
class LinearSolver{
public:

    LinearSolver(){
#if 1
        //if (!(m_ep = engOpen("matlab-8.1r2013a -nodesktop -nodisplay -nosplash -nojvm"))) {
        if (!(m_ep = engOpen("matlab -nodesktop -nodisplay -nosplash -nojvm"))) {
            fprintf(stderr, "\nCan't start MATLAB engine\n");
            exit(EXIT_FAILURE);
        }
#endif
        haveInit=false;
    }

    ~LinearSolver(){
        LOG<<"Destroying"<<endl;
        //mxDestroyArray(m_A);
        mxDestroyArray(m_result);
        //mxDestroyArray(m_b);
        engClose(m_ep);
        LOG<<"done"<<endl;
    }
    
    virtual void solve(){
        //x = exp(lsqlin(A,bm,[],[],[],[],-5*one,zer,-0.5*one,opts));
        LOG<<"SOLVING..."<<endl;
        char buffer[1024+1];
        buffer[1024] = '\0';
        engOutputBuffer(m_ep, buffer, 1024);
        //remove zero rows from A,b. quite slow!
        //engEvalString(m_ep, "zRows=~sum(A,2); A=A(~zRows,:); b=b(~zRows);");

        //engEvalString(m_ep, "lb=[-60*ones(size(A,2)/2,1);-20*ones(size(A,2)/2,1)]; ");
        //engEvalString(m_ep, "ub=[60*ones(size(A,2)/2,1);20*ones(size(A,2)/2,1)]; ");
        //engEvalString(m_ep, "lb=[-60*ones(size(A,2),1)];");
        //engEvalString(m_ep, "ub=[60*ones(size(A,2),1);]");
        engEvalString(m_ep, "lb=[-5*ones(size(A,2),1)];");
        engEvalString(m_ep, "ub=[5*ones(size(A,2),1);]");

        engEvalString(m_ep, "size(b)");
        printf("%s", buffer+2);
        engEvalString(m_ep, "size(A)");
        printf("%s", buffer+2);


        //TIME(engEvalString(m_ep, "tic;[x resnorm residual] =lsqlin(A,b,[],[],[],[],zer);toc"));
        //TIME(engEvalString(m_ep, "tic;[x resnorm residual] =lsqlin(A,b);toc"));
        engEvalString(m_ep, "options=optimset(optimset('lsqlin'),'Display','iter');");
        
        if (haveInit){
            LOGV(3)<<"solving with initial values"<<endl;
            engEvalString(m_ep, "size(init)");
            printf("%s", buffer+2);

            TIME(engEvalString(m_ep, "tic;[x resnorm residual flag lambda output] =lsqlin(A,b,[],[],[],[],lb,ub,init,options);toc"));
        }
        else{
            LOGV(3)<<"solving without initial values"<<endl;
            TIME(engEvalString(m_ep, "tic;[x resnorm residual flag lambda output] =lsqlin(A,b,[],[],[],[],lb,ub,[],options);toc"));
        }
        //TIME(engEvalString(m_ep, "tic; x=A\\b ; toc"));
        printf("%s", buffer+2);
         
        //engEvalString(this->m_ep,"clear A b init lb ub ;" );


        engEvalString(m_ep, " lambda.algorithm");
        printf("%s", buffer+2);
        engEvalString(m_ep, " resnorm");
        printf("%s", buffer+2);
        engEvalString(m_ep, " size(x)");
        printf("%s", buffer+2);

        LOG<<"done"<<endl;
        if ((m_result = engGetVariable(m_ep,"x")) == NULL)
            printf("something went wrong when getting the variable.\n Result is probably wrong. \n");
        //if ((m_residual = engGetVariable(m_ep,"residual;")) == NULL)
        //  printf("something went wrong when getting the variable residual.\n Result is probably wrong. \n");
    }
    
     void reSolve(){
        //x = exp(lsqlin(A,bm,[],[],[],[],-5*one,zer,-0.5*one,opts));
        LOG<<"SOLVING...again"<<endl;
        char buffer[1024+1];
        buffer[1024] = '\0';
        engOutputBuffer(m_ep, buffer, 1024);
        //remove zero rows from A,b. quite slow!
        //engEvalString(m_ep, "zRows=~sum(A,2); A=A(~zRows,:); b=b(~zRows);");
        
        engEvalString(m_ep, "lb=[-60*ones(size(A,2),1)];");
        engEvalString(m_ep, "ub=[60*ones(size(A,2),1);]");

        engEvalString(m_ep, "size(b)");
        printf("%s", buffer+2);
        engEvalString(m_ep, "size(A)");
        printf("%s", buffer+2);


        //TIME(engEvalString(m_ep, "tic;[x resnorm residual] =lsqlin(A,b,[],[],[],[],zer);toc"));
        //TIME(engEvalString(m_ep, "tic;[x resnorm residual] =lsqlin(A,b);toc"));
        engEvalString(m_ep, "options=optimset(optimset('lsqlin'),'Display','iter');");
        
        if (haveInit){
            engEvalString(m_ep, "init = x ;");
            printf("%s", buffer+2);

            TIME(engEvalString(m_ep, "tic;[x resnorm residual flag lambda output] =lsqlin(A,b,[],[],[],[],lb,ub,init,options);toc"));
        }
        else{
            TIME(engEvalString(m_ep, "tic;[x resnorm residual flag lambda output] =lsqlin(A,b,[],[],[],[],lb,ub,[],options);toc"));
        }
        //TIME(engEvalString(m_ep, "tic; x=A\\b ; toc"));
        printf("%s", buffer+2);
         
        engEvalString(this->m_ep,"clear A b init lb ub ;" );


        engEvalString(m_ep, " lambda.algorithm");
        printf("%s", buffer+2);
        engEvalString(m_ep, " resnorm");
        printf("%s", buffer+2);
        engEvalString(m_ep, " size(x)");
        printf("%s", buffer+2);

        LOG<<"done"<<endl;
        if ((m_result = engGetVariable(m_ep,"x")) == NULL)
            printf("something went wrong when getting the variable.\n Result is probably wrong. \n");
        //if ((m_residual = engGetVariable(m_ep,"residual;")) == NULL)
        //  printf("something went wrong when getting the variable residual.\n Result is probably wrong. \n");
    }
    
    virtual void createSystem()=0;
    //virtual void storeResult(string directory)=0;
protected:
    mxArray *m_A, *m_result,*m_b,*m_residual;
    Engine *m_ep;
    bool haveInit;
};
