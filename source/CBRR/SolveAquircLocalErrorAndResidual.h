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
class AquircLocalErrorandResidual3Solver: public AquircGlobalDeformationNormSolverCVariables< ImageType>{
public:
    typedef typename  TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef typename  DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename ImageUtils<ImageType>::FloatImagePointerType FloatImagePointerType;
    typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
    typedef typename itk::ImageRegionIterator<FloatImageType> FloatImageIterator;
    typedef typename itk::ImageRegionIteratorWithIndex<DeformationFieldType> DeformationFieldIterator;
    typedef typename DeformationFieldType::PixelType DeformationType;
    typedef typename DeformationFieldType::IndexType IndexType;
    typedef typename DeformationFieldType::PointType PointType;
    typedef typename ImageType::Pointer ImagePointerType;
    typedef typename ImageType::RegionType RegionType;
    static const unsigned int D=ImageType::ImageDimension;
public:

    virtual void SetVariables(std::vector<string> * imageIDList, map< string, map <string, DeformationFieldPointerType> > * deformationCache, map< string, map <string, DeformationFieldPointerType> > * trueDeformations,ImagePointerType ROI){
        m_imageIDList=imageIDList;
        m_deformationCache=deformationCache;
        m_numImages=imageIDList->size();
        this->m_ROI=ROI;
        if (!ROI.IsNotNull()){
            this->m_ROI=FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::createEmpty(TransfUtils<ImageType>::computeLocalDeformationNorm((*m_deformationCache)[(*m_imageIDList)[0]][(*m_imageIDList)[1]],1.0)));
        }
        m_nPixels=2* this->m_ROI->GetLargestPossibleRegion().GetNumberOfPixels( );
        m_nEqs= m_numImages*(m_numImages-1)*( m_numImages-2)*m_nPixels;
        m_nVars= m_numImages*(m_numImages-1)*m_nPixels*2;
        m_nNonZeroes=9*m_nEqs;

        //times 2 since we're now also constraining the def
        m_nEqs*=2;

        m_trueDeformations=trueDeformations;


        m_regionOfInterest.SetSize(this->m_ROI->GetLargestPossibleRegion().GetSize());
        IndexType startIndex,nullIdx;
        nullIdx.Fill(0);
        PointType startPoint;
        this->m_ROI->TransformIndexToPhysicalPoint(nullIdx,startPoint);
        (*m_deformationCache)[(*m_imageIDList)[0]][(*m_imageIDList)[1]]->TransformPhysicalPointToIndex(startPoint,startIndex);
        m_regionOfInterest.SetIndex(startIndex);

    }
    
    virtual void createSystem(){

        LOG<<"Creating equation system.."<<endl;
        LOG<<VAR(m_numImages)<<" "<<VAR(m_nPixels)<<" "<<VAR(m_nEqs)<<" "<<VAR(m_nVars)<<" "<<VAR(m_nNonZeroes)<<endl;
        mxArray *mxX=mxCreateDoubleMatrix(m_nNonZeroes,1,mxREAL);
        mxArray *mxY=mxCreateDoubleMatrix(m_nNonZeroes,1,mxREAL);
        mxArray *mxV=mxCreateDoubleMatrix(m_nNonZeroes,1,mxREAL);
        mxArray *mxB=mxCreateDoubleMatrix(m_nEqs,1,mxREAL);
        if ( !mxX || !mxY || !mxV || !mxB){
            LOG<<"couldn't allocate memory!"<<endl;
            exit(0);
        }
        double * x=( double *)mxGetData(mxX);
        std::fill(x,x+m_nNonZeroes,m_nEqs);
        double * y=( double *)mxGetData(mxY);
        std::fill(y,y+m_nNonZeroes,m_nVars);
        double * v=( double *)mxGetData(mxV);
        double * b=mxGetPr(mxB);
        
        LOG<<"creating"<<endl;
     


        char buffer[256+1];
        buffer[256] = '\0';
        engOutputBuffer(this->m_ep, buffer, 256);
      
        //attention matlab index convention?!?
        long int eq = 1;
        long int c=0;
        long int maxE=0;

       

        for (int s = 0;s<m_numImages;++s){                            
            int source=s;
            for (int i=0;i<m_numImages;++i){
                if (i!=s){
                    int intermediate=i;
                    DeformationFieldPointerType d1=(*m_deformationCache)[(*m_imageIDList)[source]][(*m_imageIDList)[intermediate]];

                    for (int t=0;t<m_numImages;++t){
                        if (t!=i && t!=s){
                            //define a set of 3 images
                            int target=t;
                            DeformationFieldPointerType d2=(*m_deformationCache)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]];
                            DeformationFieldPointerType d3=(*m_deformationCache)[(*m_imageIDList)[target]][(*m_imageIDList)[source]];
                            
                            DeformationFieldPointerType hatd1,hatd2,hatd3;
                            DeformationType hatDelta3;
                            if (m_trueDeformations!=NULL){
                                hatd1=(*m_trueDeformations)[(*m_imageIDList)[source]][(*m_imageIDList)[intermediate]];
                                hatd2=(*m_trueDeformations)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]];
                                hatd3=(*m_trueDeformations)[(*m_imageIDList)[target]][(*m_imageIDList)[source]];
                            }
                            
                            //compute circle
                            DeformationFieldPointerType circle=composeDeformations(d1,d2,d3);
                            //compute norm
                            DeformationFieldIterator it(circle,m_regionOfInterest);//this->m_ROI->GetLargestPossibleRegion());
                            it.GoToBegin();
                            
                            // LOG<<VAR(dir)<<" "<<VAR(start)<<endl;
                            for (;!it.IsAtEnd();++it){

                                bool valid=true;
                                IndexType idx3=it.GetIndex(),idx2,idx1;
                                LOGV(6)<<VAR(idx3)<<endl;
                                PointType pt1,pt2,pt3;
                                IndexType roiIdx1,roiIdx2,roiIdx3;
                                d3->TransformIndexToPhysicalPoint(idx3,pt3);
                                DeformationType delta3=d3->GetPixel(idx3);
                                hatDelta3 = hatd3->GetPixel(idx3);
                                this->m_ROI->TransformPhysicalPointToIndex(pt3,roiIdx3);

                                LOGV(6)<<VAR(idx3)<<" "<<VAR(roiIdx3)<<endl;

#if 1                       
                                //This is the backward assumption. circle errors are in the domain of d3, and are summed backwards
                                
                          
                                pt2=pt3+delta3;
                                
                                //pt2=pt3+hatd3->GetPixel(idx3);
                                d2->TransformPhysicalPointToIndex(pt2,idx2);
                                this->m_ROI->TransformPhysicalPointToIndex(pt2,roiIdx2);
                                // what to do when circle goes outside along the way?
                                // skip it
                                if ( !this->m_ROI->GetLargestPossibleRegion().IsInside(roiIdx2) ) {
                                    LOGV(6)<<"break at "<<VAR(eq)<<" "<<VAR(c)<<" "<<VAR(idx3)<<" "<<VAR(idx2)<<endl;
                                    //eq+=D;
                                    //c+=3*D;
                                    continue;
                                }
                                pt1=pt2+d2->GetPixel(idx2);
                                this->m_ROI->TransformPhysicalPointToIndex(pt1,roiIdx1);
                                //pt1=pt2+hatd2->GetPixel(idx2);
                                d1->TransformPhysicalPointToIndex(pt1,idx1);
                                if ( (!this->m_ROI->GetLargestPossibleRegion().IsInside(roiIdx1) )) {
                                    LOGV(6)<<"break at "<<VAR(eq)<<" "<<VAR(c)<<" "<<VAR(idx3)<<" "<<VAR(idx2)<<endl;
                                     //eq=eq+D;
                                    //c+=3*D;
                                    continue;
                                }
#else
                                //fixed point estimation
                                idx1=idx3;
                                idx2=idx3;
                                roiIdx1=roiIdx3;
                                roiIdx2=roiIdx3;
                                
#endif
                                
                                double val=1;

                                //add 1 for matlab array layout
                                long int e1=edgeNum(source,intermediate,roiIdx1)+1;
                                long int e2=edgeNum(intermediate,target,roiIdx2)+1;
                                long int e3=edgeNum(target,source,roiIdx3)+1;
                                if (e1<=0) {LOG<<VAR(e1)<<" ????? "<<endl;}
                                if (e2<=0) {LOG<<VAR(e2)<<endl;}
                                if (e3<=0) {LOG<<VAR(e3)<<endl; }
                                //LOG<<VAR(e1)<<" "<<VAR(e2)<<" "<<VAR(e3)<<endl;
                                
                                DeformationType localDef=it.Get();
                                
                                PointType pt0;
                                pt0=pt1+d1->GetPixel(idx1);
                                
                                LOGV(4)<<"consistency check : "<<VAR(localDef)<<" ?= "<<VAR(pt0-pt3)<<endl;
                                
                                maxE=max(maxE,max(e1,max(e2,e3)));
                                
                                for (unsigned int d=0;d<D;++d){
                                    double def=localDef[d];
                                    LOGV(6)<<VAR(e1)<<" "<<VAR(e2)<<" "<<VAR(e3)<<" "<<VAR(eq)<<" "<<VAR(c)<<endl;
                                    //set sparse entries for error variables
                                    x[c]=eq;
                                    y[c]=e1+d;
                                    v[c++]=val;
                                    x[c]=eq;
                                    y[c]=e2+d;
                                    v[c++]=val;
                                    x[c]=eq;
                                    x[c]=eq;
                                    y[c]=e3+d;
                                    v[c++]=val;

                                      //set sparse entries for true deformation variables
                                    x[c]=eq;
                                    y[c]=e1+d+m_nPixels;
                                    v[c++]=val;
                                    x[c]=eq;
                                    y[c]=e2+d+m_nPixels;
                                    v[c++]=val;
                                    x[c]=eq;
                                    y[c]=e3+d+m_nPixels;
                                    v[c++]=val;
                                    //set rhs
                                   //b[eq-1]=delta3[d];
                                    b[eq-1]=def;
                                    ++eq;

                                    //set eqn for soft constraining the estimated true deformation to be similar to the original deformation
                                    float lambda=exp(-0.5*def*def/25);
                                    x[c]=eq;
                                    y[c]=e1+d+m_nPixels;
                                    v[c++]=lambda*val;  
                                    x[c]=eq;
                                    y[c]=e2+d+m_nPixels;
                                    v[c++]=lambda*val;
                                    x[c]=eq;
                                    y[c]=e3+d+m_nPixels;
                                    v[c++]=lambda*val;
                                    b[eq-1]=0.0;
                                    //b[eq-1]=lambda*hatDelta3[d];
                                    ++eq;
                                    LOGV(6)<<"did it"<<endl;
                                }// D
                            }//image

                        }//if
                    }//target
                }//if
            }//intermediate
        }//source
        LOG<<VAR(eq)<<" "<<VAR(c)<<endl;
        LOG<<VAR(maxE)<<endl;
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
        double trueResidual=0.0;
        double estimationResidual=0.0;
        double circleResidual=0.0;
        int c=0;
        for (int s = 0;s<m_numImages;++s){
            for (int t=0;t<m_numImages;++t){
                if (s!=t){
                    //slightly(!!!) stupid creation of empty image
                    DeformationFieldPointerType estimatedError=TransfUtils<ImageType>::createEmpty(this->m_ROI);//ImageUtils<DeformationFieldType>::createEmpty((*m_deformationCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]]);
                    DeformationFieldIterator itErr(estimatedError,estimatedError->GetLargestPossibleRegion());
                    DeformationFieldPointerType estimatedDeform=TransfUtils<ImageType>::createEmpty(this->m_ROI);//ImageUtils<DeformationFieldType>::createEmpty((*m_deformationCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]]);
                    DeformationFieldIterator itDef(estimatedDeform,estimatedDeform->GetLargestPossibleRegion());
                    DeformationFieldIterator itTrueDef((*m_trueDeformations)[(*m_imageIDList)[s]][(*m_imageIDList)[t]],m_regionOfInterest);
                    DeformationFieldIterator itOriginalDef((*m_deformationCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]],m_regionOfInterest);
                    itErr.GoToBegin();
                    itDef.GoToBegin();
                    itTrueDef.GoToBegin();
                    itOriginalDef.GoToBegin();
                    for (int p=0;!itErr.IsAtEnd();++itErr,++itDef,++itTrueDef,++itOriginalDef){
                        //get solution of eqn system
                        DeformationType dispErr,dispDef;
                        int e=edgeNum(s,t,itErr.GetIndex());
                        for (unsigned int d=0;d<D;++d,++p){
                            dispErr[d]=rData[e+d];
                            dispDef[d]=rData[e+d+m_nPixels];
                        }
                        itErr.Set(dispErr);
                        itDef.Set(dispDef);
                        
                        //compute errors
                        //1. compute derivation of assumption that estimatedError+estimatedDeform=originalDeform
                        estimationResidual+=(dispErr+dispDef-itOriginalDef.Get()).GetSquaredNorm();
                        //2. compute difference of estimated deform and true deform
                        trueResidual+=(itTrueDef.Get()-dispDef).GetSquaredNorm();
                        ++c;
                    }

                    ostringstream outfile;
                    outfile<<directory<<"/estimatedLocalComposedDeformationError-FROM-"<<(*m_imageIDList)[s]<<"-TO-"<<(*m_imageIDList)[t]<<".mha";
                    ImageUtils<DeformationFieldType>::writeImage(outfile.str().c_str(),estimatedError);
                    ostringstream outfile2;
                    outfile2<<directory<<"/estimatedLocalComposedDeformation-FROM-"<<(*m_imageIDList)[s]<<"-TO-"<<(*m_imageIDList)[t]<<".mha";
                    ImageUtils<DeformationFieldType>::writeImage(outfile2.str().c_str(),estimatedDeform);
                }
            }
        }
        estimationResidual=sqrt(estimationResidual);
        trueResidual=sqrt(trueResidual);
        LOG<<VAR(estimationResidual)<<" "<<VAR(trueResidual)<<" "<<VAR(c)<<endl;
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
    map< string, map <string, DeformationFieldPointerType> > * m_deformationCache,* m_trueDeformations;
    std::vector<string> * m_imageIDList;
    bool m_additive;
    RegionType m_regionOfInterest;

protected:
    //return fortlaufende number of pairs n1,n2, 0..(n*(n-1)-1)
    inline long int edgeNum(int n1,int n2){ return ((n1)*(m_numImages-1) + n2 - (n2>n1));}
    
    //return edgenumber after taking into acount nPixel*2 edges per image pair
    inline long int edgeNum(int n1,int n2,IndexType idx){ 
        long int offset = this->m_ROI->ComputeOffset(idx);
        return offset*2+edgeNum(n1,n2)*m_nPixels*2 ;
    }
  

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
