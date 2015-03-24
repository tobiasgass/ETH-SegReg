#/bin/bash

binDir=/Users/tobi/Arbeit/src/ETH-SegReg/build/bin
dataDir=/Users/tobi/Arbeit/src/ETH-SegReg/data

if [ ! -e $dataDir/List.IDs ]
then
    bash $dataDir/prepareData.sh
fi

N=`wc -l $dataDir/List.IDs | awk '{print $1}'`

mkdir -p MRegFuseData/inputDeformations
mkdir -p MRegFuseData/outputDeformations

##generate random pairwise registrations
echo -n "" >MRegFuseData/pairwiseDeformations.List
for n1 in `seq  1 $N`
do
    for n2 in `seq  1 $N | grep -v $n1`
    do
	#random, doesnt work well
	#$binDir/GenerateDeformation2D --target $dataDir/Images/img-$n2.png --out MRegFuseData/inputDeformations/def-$n1-$n2.mha --linear
	$binDir/SRS2D-Bone --t $dataDir/Images/img-$n2.png --a $dataDir/Images/img-$n1.png --sa $dataDir/Segmentations/seg-$n1.png --ta /dev/null/tmp.nii --tsa /dev/null/tno.nii --T MRegFuseData/inputDeformations/def-$n1-$n2.mha 
	
	echo "$n1 $n2 `pwd`/MRegFuseData/inputDeformations/def-$n1-$n2.mha" >>MRegFuseData/pairwiseDeformations.List
    done
done

