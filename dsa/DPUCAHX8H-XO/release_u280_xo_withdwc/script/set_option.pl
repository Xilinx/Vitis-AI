#! /usr/bin/env perl
#use strict ; 
use warnings ;
use 5.010 ;

my $version = "1.0";

use Getopt::Long;           # GetOptions
use File::Basename;         # fileparse, basename, dirname
use File::stat;             # ->mtime
use Pod::Usage;             # pod2usage

my $DPU_ARCH = "DPU_V3E";
my $option_list ;#option_list.u50 option_list.u50lv option_list.u280
my $DPU_GIT;#
my $debug = "false";

GetOptions( 'DPU_ARCH=s' => \$DPU_ARCH,
	    'DPU_GIT=s' => \$DPU_GIT,
	    'option_list=s' => \$option_list,
	    'debug=s' => \$debug,);

if ($DPU_ARCH eq "DPU_V4E" or $DPU_ARCH eq "DPUCVDX8H") {
    $top = "$DPU_GIT/ip/dpu/rtl/cnn/DPUCVDX8H.v";
} elsif ($DPU_ARCH eq "DPU_V3E") {
    $top = "$DPU_GIT/ip/dpu/rtl/cnn/dpu_top.v";
} elsif ($DPU_ARCH eq "DPU_V3ME") {
    $top = "$DPU_GIT/ip/dpu/rtl/cnn/DPUCAHX8L.v";
} else {
    print "Error: Not recognized architecture of DPU - $DPU_ARCH\n"
}

open(TOP, $top);
@new_top=<TOP>;
foreach(@new_top)
{
	$line=$_;
	#print $_ ;
		if(/\s*parameter\s*(\S*)/)
		{$para_name=$1;
#				print $1."\n";
		open(OPTION,$option_list) or print "FATAL: Can't open option file - $option_list for $!\n";
		while(<OPTION>)
			{
			if(/(\S*)\s*(\S*)/)
			{$set_value=$2;}
			#		print $_;
			#		pr	print $2;
			if($1 eq $para_name)
				{
				$line =~ s/(parameter\s*\S*\s*=\s*)\S*/$1$set_value/ ;
				last;
				}
			}
			# print "$_ \n";
		 }
		 $_ =$line;
}
close TOP;
open (TOP, ">", $top);
print TOP @new_top;
close TOP;


