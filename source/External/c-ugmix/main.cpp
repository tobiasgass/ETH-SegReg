#include <stdio.h>
#include <iostream>
#include <fstream>
#include "unsupervised.h"
using namespace std;

#define Dimension 1
#define SAMPLE 3000

int main(){
  ifstream fin;
  char str[80];
  float tmp = 0.0;
  
  /*define the model*/
  unsupervised model;
  
  Matrix obs(Dimension,SAMPLE);

  for(int i=0;i<3;i++){

    sprintf_s(str, "./test%d.dat", i);
    fin.open(str);
    
    if(!fin){
      cout << "Can't open input file.\n";
      return 1;
    }
    
    for(int j=0;j<1000;j++){
      fin >> tmp;
      if(!i) tmp += 5.0;
      else if(i==1) tmp += 10.0;
      
      obs.element(0,i*1000+j) = tmp;
    }
    fin.close();
  }
  
  model.estimate(20,obs);
  model.display();

  obs.CleanUp();
  model.CleanUp();
  
  return 0;
}
 
