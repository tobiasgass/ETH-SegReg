
#include "ExplicitInstantiationsPotentials.h"
#include "itkImage.h"

namespace SRS{

  template class PotentialInstantiations<itk::Image<short,2> >;
  template class PotentialInstantiations<itk::Image<float,2> >;
  template class PotentialInstantiations<itk::Image<double,2> >;
  template class PotentialInstantiations<itk::Image<int,2> >;
  template class PotentialInstantiations<itk::Image<unsigned char,2> >;
  template class PotentialInstantiations<itk::Image<unsigned short,2> >;
 
  template class PotentialInstantiations<itk::Image<short,3> >;
  template class PotentialInstantiations<itk::Image<float,3> >;
  template class PotentialInstantiations<itk::Image<double,3> >;
  template class PotentialInstantiations<itk::Image<int,3> >;
  template class PotentialInstantiations<itk::Image<unsigned char,3> >;
  template class PotentialInstantiations<itk::Image<unsigned short,3> >;
 



}//namespace
