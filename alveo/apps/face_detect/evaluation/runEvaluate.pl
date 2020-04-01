#!/usr/bin/perl -w

use strict;

#### VARIABLES TO EDIT ####
# where gnuplot is
my $GNUPLOT = "/sw/bin/gnuplot"; 
# where the binary is
my $evaluateBin = "evaluate"; 
# where the images are
my $imDir = "facesInTheWild/"; 
# where the folds are
my $fddbDir = "FDDB-folds"; 
# where the detections are
my $detDir = "yourDetectionOutputDirectory"; 
###########################

my $detFormat = 0; # 0: rectangle, 1: ellipse 2: pixels

sub makeGNUplotFile
{
  my $rocFile = shift;
  my $gnuplotFile = shift;
  my $title = shift;
  my $pngFile = shift;

  open(GF, ">$gnuplotFile") or die "Can not open $gnuplotFile for writing\n"; 
  #print GF "$GNUPLOT\n";
  print GF "set term png\n";
  print GF "set size .75,1\n";
  print GF "set output \"$pngFile\"\n";
  #print GF "set xtics 500\n";
  #print GF "set logscale x\n";
  print GF "set ytics .1\n";
  print GF "set grid\n";
  #print GF "set size ratio -1\n";
  print GF "set ylabel \"True positive rate\"\n";
  print GF "set xlabel \"False positives\"\n";
  #print GF "set xr [0:50000]\n";
  print GF "set yr [0:1]\n";
  print GF "set key right bottom\n";
  print GF "plot \"$rocFile\" using 2:1 with linespoints title \"$title\"\n";
  close(GF);
}

my $annotFile = "ellipseList.txt";
my $listFile = "imList.txt";
my $gpFile = "createROC.p";

# read all the folds into a single file for evaluation
my $detFile = $detDir;
$detFile =~ s/\//_/g;
$detFile = $detFile."Dets.txt";

if(-e $detFile){
  system("rm", $detFile);
}

if(-e $listFile){
  system("rm", $listFile);
}

if(-e $annotFile){
  system("rm", $annotFile);
}

foreach my $fi (1..10){
  my $foldFile = sprintf("%s/fold-%02d-out.txt", $detDir, $fi);
  system("cat $foldFile >> $detFile");
  $foldFile = sprintf("%s/FDDB-fold-%02d.txt", $fddbDir, $fi);
  system("cat $foldFile >> $listFile");
  $foldFile = sprintf("%s/FDDB-fold-%02d-ellipseList.txt", $fddbDir, $fi);
  system("cat $foldFile >> $annotFile");
}

#die;
# run the actual evaluation code to obtain different points on the ROC curves
#system($evaluateBin, "-a", $annotFile, "-d", $detFile, "-f", $detFormat, "-i", $imDir, "-l", $listFile, "-r", $detDir, "-s");
system($evaluateBin, "-a", $annotFile, "-d", $detFile, "-f", $detFormat, "-i", $imDir, "-l", $listFile, "-r", $detDir);

# plot the two ROC curves using GNUplot
makeGNUplotFile($detDir."ContROC.txt", $gpFile, $detDir, $detDir."ContROC.png");
system("echo \"load '$gpFile'\" | $GNUPLOT");

makeGNUplotFile($detDir."DiscROC.txt", $gpFile, $detDir, $detDir."DiscROC.png");
system("echo \"load '$gpFile'\" | $GNUPLOT");

# remove intermediate files
system("rm", $annotFile, $listFile, $gpFile, $detFile);
