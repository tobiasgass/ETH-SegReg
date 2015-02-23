#pragma once
#include "matrix.h"
#include "BaseLinearSolver.h"
#include "TransformationUtils.h"
#include "Log.h"
#include "ImageUtils.h"
#include <vector>
#include <sstream>
#include "SolveAquircGlobalDeformationNormCVariables.h"


template<class ImageType>
class AquircLocalDeformationNormSolver: public AquircGlobalDeformationNormSolverCVariables< ImageType>{
public:
    typedef typename  TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef typename  DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename ImageUtils<ImageType>::FloatImagePointerType FloatImagePointerType;
    typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
    typedef typename itk::ImageRegionIterator<typename ImageUtils<ImageType>::FloatImageType> FloatImageIterator;
public:

    virtual void SetVariables(std::vector<string> * imageIDList, map< string, map <string, DeformationFieldPointerType> > * deformationCache, map< string, map <string, DeformationFieldPointerType> > * trueDeformations=NULL){
        m_imageIDList=imageIDList;
        m_deformationCache=deformationCache;
        m_numImages=imageIDList->size();
        m_nPixels=(*deformationCache)[(*imageIDList)[0]][(*imageIDList)[1]]->GetLargestPossibleRegion().GetNumberOfPixels( );
        m_nEqs= m_numImages*(m_numImages-1)*( m_numImages-2)*m_nPixels;
        m_nVars= m_numImages*(m_numImages-1)*m_nPixels;
        m_nNonZeroes=3*m_nEqs;
    }
    
    virtual void createSystem(){

        LOG<<"Creating equation system.."<<endl;
        LOG<<VAR(m_numImages)<<" "<<VAR(m_nPixels)<<" "<<VAR(m_nEqs)<<" "<<VAR(m_nVars)<<" "<<VAR(m_nNonZeroes)<<endl;
        mxArray *mxX=mxCreateDoubleMatrix(m_nNonZeroes,1,mxREAL);
        mxArray *mxY=mxCreateDoubleMatrix(m_nNonZeroes,1,mxREAL);
        mxArray *mxV=mxCreateDoubleMatrix(m_nNonZeroes,1,mxREAL);
        mxArray *mxB=mxCreateDoubleMatrix(m_nEqs,1,mxREAL);
        
        double * x=( double *)mxGetData(mxX);
        double * y=( double *)mxGetData(mxY);
        double * v=( double *)mxGetData(mxV);
        double * b=mxGetPr(mxB);
        
        

        //m_A = mxCreateSparse(m_nVars, m_nEqs, m_nNonZeroes);
        //engPutVariable(this->m_ep, "A", m_A);
        LOG<<"creating"<<endl;
        {
            ostringstream evalstr;
            evalstr<<"A = sparse([],[],[],"<<m_nEqs<<","<<m_nVars<<","<<m_nNonZeroes<<");";
            engEvalString(this->m_ep,evalstr.str().c_str() );
        }

        {
            ostringstream evalstr;
            evalstr<<" b=zeros(1,"<<m_nEqs<<");";
            engEvalString(this->m_ep,evalstr.str().c_str() );
        }


        char buffer[256+1];
        buffer[256] = '\0';
        engOutputBuffer(this->m_ep, buffer, 256);
      
        //attention matlab index convention?!?
        int eq = 1;
        //create edge index storage object
        int c=0;
        for (int s = 0;s<m_numImages;++s){
            for (int i=s+1;i<m_numImages;++i){
                for (int t=i+1;t<m_numImages;++t){
                    //define a set of 3 images
                    //there are 6 possible circles with 3 images
                    int source=s;
                    int intermediate=i;
                    int target=t;
                    //       LOG<<VAR(s)<<" "<<VAR(i)<<" "<<VAR(t)<<endl;
                    for (int dir=0;dir<2;++dir){ //forwar-backward
                        double normSum=0.0;
                        for (int start=0;start<3;++start){
                            if (intermediate<0 || target<0 || intermediate>m_numImages || target>m_numImages) break;

                            //compute circle
                            DeformationFieldPointerType circle=composeDeformations((*m_deformationCache)[(*m_imageIDList)[source]][(*m_imageIDList)[intermediate]],
                                                                                   (*m_deformationCache)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]],
                                                                                   (*m_deformationCache)[(*m_imageIDList)[target]][(*m_imageIDList)[source]]);
                            //compute norm
                            //double norm=TransfUtils<ImageType>::computeDeformationNorm(circle,1.0);
                            FloatImagePointerType norm=TransfUtils<ImageType>::computeLocalDeformationNorm(circle,1.0);
                            FloatImageIterator it(norm,norm->GetLargestPossibleRegion());
                            it.GoToBegin();
                            for (int p=0;!it.IsAtEnd();++it,++p){
                                double localNorm=it.Get();
                                if (!m_additive){
                                    localNorm=max(log(localNorm),-15.0);
                                }
                                //set sparse entries
                                x[c]=eq;
                                y[c]=edgeNum(source,intermediate,p);
                                //LOG<<VAR(source)<<" "<<VAR(intermediate)<<" "<<VAR(p)<<" "<<VAR(edgeNum(source,intermediate,p))<<endl;
                                v[c++]=1;
                                x[c]=eq;
                                y[c]=edgeNum(intermediate,target,p);
                                v[c++]=1;
                                x[c]=eq;
                                y[c]=edgeNum(target,source,p);
                                v[c++]=1;
                                
                                //set rhs
                                b[eq-1]=3*localNorm;
                                ++eq;
                            }
                            
                            
                            
                            //shift start point
                            int tmpInt=source;
                            source=intermediate;
                            intermediate=target;
                            target=tmpInt;
                        }
                      
                        int tmp=target;
                        target=intermediate;
                        intermediate=tmp;

                    }

                
                }
            }
        }
        
        //put variables into workspace and immediately destroy them
        engPutVariable(this->m_ep,"xCord",mxX);
        mxDestroyArray(mxX);
        engPutVariable(this->m_ep,"yCord",mxY);
        mxDestroyArray(mxY);
        engPutVariable(this->m_ep,"val",mxV);
        mxDestroyArray(mxV);
        engPutVariable(this->m_ep,"b",mxB);
        mxDestroyArray(mxB);
        engEvalString(this->m_ep,"A=sparse(xCord,yCord,val);" );
        //clear unnneeded variables from matlab workspace
        engEvalString(this->m_ep,"clear xCord yCord val;" );

    }

    virtual void storeResult(string directory){
        std::vector<double> result(m_nVars);
        double * rData=mxGetPr(this->m_result);
        for (int s = 0;s<m_numImages;++s){
            for (int t=0;t<m_numImages;++t){
                if (s!=t){
                    //slightly(!!!) stupid creation of empty image
                    FloatImagePointerType normImage=ImageUtils<FloatImageType>::createEmpty(TransfUtils<ImageType>::computeLocalDeformationNorm((*m_deformationCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]],1.0));
                    FloatImageIterator it(normImage,normImage->GetLargestPossibleRegion());
                    it.GoToBegin();
                    for (int p=0;!it.IsAtEnd();++it,++p){
                        it.Set(50*rData[edgeNum(s,t,p)]);
                    }

                    ostringstream outfile;
                    outfile<<directory<<"/estimatedLocalDeformationNorm-from"<<(*m_imageIDList)[s]<<"-TO-"<<(*m_imageIDList)[t]<<".png";
                    ImageUtils<ImageType>::writeImage(outfile.str().c_str(),FilterUtils<FloatImageType,ImageType>::truncateCast(normImage));
                }
            }
        }

    }

    std::vector<double> getResult(){
        std::vector<double> result(m_nVars);
        double * rData=mxGetPr(this->m_result);
        for (int i=0;i<m_nVars;++i){
            result[i]=rData[i];
            int n1,n2;
            edges(i+1,n1,n2);
            LOG<<VAR(i)<<" "<<VAR(n1)<<" "<<VAR(n2)<<" "<<VAR(result[i])<<endl;
        }
        return result;
        

    }
protected:
    int m_nVars,m_nEqs,m_nNonZeroes,m_nPixels;
    int m_numImages;
    map< string, map <string, DeformationFieldPointerType> > * m_deformationCache;
    std::vector<string> * m_imageIDList;
    bool m_additive;

protected:
    inline int edgeNum(int n1,int n2,int p){ return ((n1)*(m_numImages-1) + n2 - (n2>n1))*(m_nPixels) +p +1;}
    inline void edges(int edgeNum, int &n1, int &n2){
        n1 = edgeNum/(m_numImages-1);
        n2 =edgeNum%(m_numImages-1);
        if (n2 ==0){
            n2=(m_numImages-1);
            n1--;
        }
        if (n2>n1) ++n2;
        n2--;
        
    }

    //compose 3 deformations. order is left-to-right
    DeformationFieldPointerType composeDeformations(DeformationFieldPointerType d1,DeformationFieldPointerType d2,DeformationFieldPointerType d3){
        return TransfUtils<ImageType>::composeDeformations(d3,TransfUtils<ImageType>::composeDeformations(d2,d1));

    }
};
