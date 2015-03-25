#!/bin/bash

N=4

mkdir -p DeformationFields

binDir="echo ../build/bin/"

###
#Create N random deformations and their inverse
for n in `seq  2 $N`
do
    $binDir/GenerateDeformation2D --target Images/img-1.nii --out DeformationFields/def-1-$n.mha  --linear
    #invert deformation
    $binDir/InvertDeformation2D  DeformationFields/def-1-$n.mha  DeformationFields/def-$n-1.mha 
done


#Deform images and segmentation
for n in `seq  2 $N`
do
    $binDir/DeformImage2D --moving Images/img-1.nii --def DeformationFields/def-1-$n.mha --out Images/img-$n.nii
    $binDir/DeformImage2D --moving Segmentations/seg-1.nii --def DeformationFields/def-1-$n.mha --out Segmentations/seg-$n.nii --NN

done



#Create pairwise registrations
for n1 in `seq  2 $N`
do
    for n2 in `seq  2 $N | grep -v $n1`
    do
	$binDir/ComposeDeformations2D   DeformationFields/def-$n2-1.mha DeformationFields/def-1-$n1.mha DeformationFields/def-$n2-$n1.mha 
    done
done


#Create file lists
echo -n "" > List.IDs
echo -n "" >List.Segmentations
echo -n "" >List.Images
echo -n "" >List.DeformationFields

for n in `seq 1 $N`
do
    echo $n >>List.IDs
    echo $n `pwd`/Images/img-$n.nii >>List.Images
    echo $n `pwd`/Segmentations/seg-$n.nii >>List.Segmentations
    for n2 in `seq 1 $N | grep -v $n`
    do
	echo $n $n2 `pwd`/DeformationFields/def-$n-$n2.mha >>List.DeformationFields
    done
done
