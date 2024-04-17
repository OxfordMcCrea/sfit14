#! /bin/bash
#$ -cwd
#$ -pe smp 8
#$ -l s_rt=300:00:00
#$ -l mem_free=30G
#
#Record origin directory
ORIG=`pwd`
#Set "scratch" environment
SCR=$TMPDIR
mkdir -p $SCR
#Name of the files you want to copy onto scratch node
INFILE1=ApseModelPDDCS8.dat
INFILE2=functions.py
INFILE3=iso_jp8.dat
INFILE4=ND3_Levels.dat
INFILE5=rpddcs_jp8.dat
INFILE6=s_fitparameters.py
INFILE7=s_fit14_Correct_Distribution.py
INFILE8=BackScattR00.dat
INFOLD=Functions

cp $INFILE1 $SCR
cp $INFILE2 $SCR
cp $INFILE3 $SCR
cp $INFILE4 $SCR
cp $INFILE5 $SCR
cp $INFILE6 $SCR
cp $INFILE7 $SCR
cp $INFILE8 $SCR
cp -r $INFOLD $SCR

# After we have copied files from hartree login node
cd $SCR

#run python script
python3 $INFILE7 > status.txt

# Copy everything back to the login node directory
cp * $ORIG
cp -r * $ORIG
cd $ORIG
