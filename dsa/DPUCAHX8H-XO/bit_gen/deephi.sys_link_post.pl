#! /usr/bin/env perl -w
use strict ; 
use warnings ;
use 5.010 ;

my $version = "1.0";

use Getopt::Long;           # GetOptions
use File::Basename;         # fileparse, basename, dirname
use File::stat;             # ->mtime
use Pod::Usage;             # pod2usage
use POSIX;
#use Switch;

#===================================================================================================
#sub and misc
our $iter = 0;
my $length = 0;

#Sub: locate engine's to left or right
#input: engine's number
#output: which engines are located on the rigth part; I divide the floorplan into  left and rigth parts because we only have left and rigth for hbm
#Note: left and right are not equal to stack0 or stack1. left and right are relative position, and weight and instruction port should be in the "middle"
sub locate_engine {
    my ( $eng_num ) = @_ ;

    my @left;
    my @right;
    given ($eng_num) {
        when ($_ eq "3" ) {
            @left = split (" ", "0");
            @right = split (" ", "1 2");
	}
        when ($_ eq "4" ) {
            @left = split (" ", "0 3");
            @right = split (" ", "1 2");
	}
        when ($_ eq "5" ) {
            @left = split (" ", "0 1 4");
            @right = split (" ", "2 3");
	}
        when ($_ eq "6" ) {
            @left = split (" ", "0 1 5");
            @right = split (" ", "2 3 4");
	}
        default {print "ERROR: not engine number for $eng_num\n";}
    }
    return (\@left, \@right);
}

#expand port name
sub expand_port_name {
    my ( $core_name, $pprefix, $ports ) = @_ ;

    my @full_name_ports;
    foreach (@$ports) {
	push (@full_name_ports,"/$core_name/${pprefix}$_");
    }
    return (\@full_name_ports);

}

#allocate
#allocate into result0 or result1

our @result0 = ();
our @result1 = ();
sub allocate {
    my ($init, $step, $iter_limit) = @_;
    

    if ($init >= 0) {
	push (@result1,$iter);
        $init = $init - $step;
    } else {
	push (@result0,$iter);
        $init = $init + $step;
    }

    $iter++;

    #print "Inside allocate: $iter: $init $step\n";
    if ($iter == $iter_limit) {
	return;
    } else {
	allocate($init, $step, $iter_limit);
	return;
    }
}

sub trim_DPU_AXI {
    my $str = shift ;
    $str =~ s#/dpu_[0-9]/DPU_AXI_##g ;
    return $str ;
}

#allocate(0,2,2);
#print "Result0: @result0\n";
#print "Result1: @result1\n";

#===================================================================================================
#Main

#Note: for priority FORCE_ALL_HBM_ASSIGN > MANUALLY_HBM_ASSIGN, and this script will pick up the ports from MANUALLY_HBM_ASSIGN and let each port to access all the MCs
my $SLR0 = 0 ;#engine number
my $SLR1 = 0 ;#engine number
my $SLR2 = 0 ;#engine number
my $FORCE_D_LOC = "balance";#"balance" - or "0" - stack0 or "1" - stack1; force FeatureMap into special location or not
my $FORCE_W_LOC = "balance";#"balance" - or "0" - stack0 or "1" - stack1; force the Weight into special location or not
my $FORCE_I_LOC = "balance";#"balance" - or "0" - stack0 or "1" - stack1; force the Instruction special location or not
my $ARCH = "v3e";#"v3e or v3me
my $ALVEO;
my $SYS_LINK_POST = "./bit_gen/constraints/sys_link_post.tcl";
my $FORCE_ALL_HBM_ASSIGN = "false";# defautly, each kernel's AXI port can only access matching MC. You can use this option to let each AXI port access all HBM's MC
my $TXT = "hbm_assignment.txt";
my $MANUALLY_HBM_ASSIGN = "";# if you have a hacked TXT by manually, this scripts will override the HBM MC allocation
my $AXI_SHARE = "false";#if we merge W0 W1 I into I
my $HBM_FREQ = 600;#currently, this option is only used for u50 card
my $debug = "false";
my $u50_rmvmss = "false";
my $w_cross_hbm_switch = "false";#because normally there will be 2 or more weights ports and they should be in the same HBM's MC - not cross switch
my $host_port_alloc  ;
my $shell_hier  ;
my $DEVICE = "xilinx_u280_xdma_201920_1";
GetOptions( 'SLR0=i' => \$SLR0,
	    'SLR1=i' => \$SLR1,
	    'SLR2=i' => \$SLR2,
	    'FORCE_D_LOC=s' => \$FORCE_D_LOC,
	    'FORCE_W_LOC=s' => \$FORCE_W_LOC,
	    'FORCE_I_LOC=s' => \$FORCE_I_LOC,
	    'ARCH=s' => \$ARCH,
	    'ALVEO=s' => \$ALVEO,
	    'SYS_LINK_POST=s' => \$SYS_LINK_POST,
	    'FORCE_ALL_HBM_ASSIGN=s' => \$FORCE_ALL_HBM_ASSIGN,
	    'TXT=s' => \$TXT,
	    'MANUALLY_HBM_ASSIGN=s' => \$MANUALLY_HBM_ASSIGN,
	    'AXI_SHARE=s' => \$AXI_SHARE,
	    'HBM_FREQ=i' => \$HBM_FREQ,
	    'DEVICE=s' => \$DEVICE,
	    'debug=s' => \$debug,);
if ( $ALVEO eq "u280" ){
$host_port_alloc=8;}
elsif ( $ALVEO eq "u50" or $ALVEO eq "u50lv"  ){
$host_port_alloc=28;}
else {
$host_port_alloc=8;}

if ( $ALVEO eq "u280" ){
$shell_hier="pfm_top_i/dynamic_region";}
else {
$shell_hier="level0_i/ulp";}


my $pprefix = "";#port's prefix
my $psuffix = "";#port's suffix
my @ports;
if ($ARCH eq "v3e") {
    $pprefix = "DPU_AXI_";
    @ports = qw(0 1 2 4 5 I0 W0 W1);
}

#collect ports
my @SLR0_W_PORT;
my @SLR0_I_PORT;
my @SLR0_D_PORT;
if ( $SLR0 ne 0 ) { 
    @SLR0_W_PORT = split(" ", "/dpu_0${pprefix}W0 /dpu_0${pprefix}W1");
    @SLR0_I_PORT = split(" ", "/dpu_0${pprefix}I0");
    for ($iter=0; $iter<$SLR0; $iter++) {
        push (@SLR0_D_PORT, "/dpu_0${pprefix}$iter");
    }
}

my @SLR1_W_PORT;
my @SLR1_I_PORT;
my @SLR1_D_PORT;
if ( $SLR1 ne 0 ) { 
    @SLR1_W_PORT = split(" ", "/dpu_1${pprefix}W0 /dpu_1${pprefix}W1");
    @SLR1_I_PORT = split(" ", "/dpu_1${pprefix}I0");
    for ($iter=0; $iter<$SLR1; $iter++) {
        push (@SLR1_D_PORT, "/dpu_1${pprefix}$iter");
    }
}

my @SLR2_W_PORT;
my @SLR2_I_PORT;
my @SLR2_D_PORT;
if ( $SLR2 ne 0 ) { 
    @SLR2_W_PORT = split(" ", "/dpu_2${pprefix}W0 /dpu_2${pprefix}W1");
    @SLR2_I_PORT = split(" ", "/dpu_2${pprefix}I0");
    for ($iter=0; $iter<$SLR2; $iter++) {
        push (@SLR2_D_PORT, "/dpu_2${pprefix}$iter");
    }
}

#===================================================================================================
#Allocate Strategy
my $current_stage = "D";
my @slr_encode;#record which SLRs are deployed
#1.1 seperate data by stack 0 and stack 1:
#	1.1.1 stack 0's port: start from port 0 to port 15
#	1.1.2 stack 1's port: start from port 16 to port 31
#1.2 seperate data by stack 0 and stack 1. However, two stategy for weights
#	1.2.1: make weights next to data's ports - currently
#	1.2.2: make weights close to the "central" - stack 0's weight ports start from port 15 to port 0; stack 1's weight ports start from 16 to 31(next to data's ports)
#1.3 seperate data by stack 0 and stack 1. same as weights

#1.1 allocate data(feature map) ports
my @HBM_STACK0_ALLOC;
my @HBM_STACK1_ALLOC;
#my @HBM_STACK0_ALLOC_4SLR0;
#my @HBM_STACK1_ALLOC_4SLR0;
#my @HBM_STACK0_ALLOC_4SLR1;
#my @HBM_STACK1_ALLOC_4SLR1;
#my @HBM_STACK0_ALLOC_4SLR2;
#my @HBM_STACK1_ALLOC_4SLR2;
my @HBM_STACK0_ALLOC_W;
my @HBM_STACK1_ALLOC_W;
my @HBM_STACK0_ALLOC_I;
my @HBM_STACK1_ALLOC_I;
my @HBM_STACK0_ALLOC_D;
my @HBM_STACK1_ALLOC_D;
my @HBM_LEFT_ALLOC_D;
my @HBM_RIGHT_ALLOC_D;

my $left;
my $right;
my $left_ports;
my $right_ports;
if ($SLR0 ne "0") {
    ($left,$right) = locate_engine($SLR0);
    $left_ports = expand_port_name("dpu_0",$pprefix,$left);
    $right_ports = expand_port_name("dpu_0",$pprefix,$right);
    push (@HBM_LEFT_ALLOC_D,@$left_ports);
    push (@HBM_RIGHT_ALLOC_D,@$right_ports);
    push (@slr_encode,"dpu_0");
}
if ($SLR1 ne "0") {
    ($left,$right) = locate_engine($SLR1);
    $left_ports = expand_port_name("dpu_1",$pprefix,$left);
    $right_ports = expand_port_name("dpu_1",$pprefix,$right);
    push (@HBM_LEFT_ALLOC_D,@$left_ports);
    push (@HBM_RIGHT_ALLOC_D,@$right_ports);
    push (@slr_encode,"dpu_1");
}
if ($SLR2 ne "0") {
    ($left,$right) = locate_engine($SLR2);
    $left_ports = expand_port_name("dpu_2",$pprefix,$left);
    $right_ports = expand_port_name("dpu_2",$pprefix,$right);
    push (@HBM_LEFT_ALLOC_D,@$left_ports);
    push (@HBM_RIGHT_ALLOC_D,@$right_ports);
    push (@slr_encode,"dpu_2");
}

#locate feature map's ports
given ($FORCE_D_LOC) {
    when ($_ eq "balance") {
	@HBM_STACK0_ALLOC_D = split(",",join(",",@HBM_LEFT_ALLOC_D));
	@HBM_STACK1_ALLOC_D = split(",",join(",",@HBM_RIGHT_ALLOC_D));
    }
    when ($_ eq "0") {
	@HBM_STACK0_ALLOC_D = split(",",join(",",@HBM_LEFT_ALLOC_D));
        push (@HBM_STACK0_ALLOC_D, split(",",join(",",@HBM_RIGHT_ALLOC_D)));
    }
    when ($_ eq "1") {
	@HBM_STACK1_ALLOC_D = split(",",join(",",@HBM_LEFT_ALLOC_D));
        push (@HBM_STACK1_ALLOC_D, split(",",join(",",@HBM_RIGHT_ALLOC_D)));
    }
    default {
	print "ERROR: not recognized options for FORCE_D_LOC - $FORCE_D_LOC\n";
    }
}

if ( $debug eq "true" ) {print "Current D allocationat stack0: @HBM_STACK0_ALLOC_D\n";}
if ( $debug eq "true" ) {print "Current D allocationat stack1: @HBM_STACK1_ALLOC_D\n";}
#@HBM_STACK0_ALLOC = split(",",join(",",@HBM_STACK0_ALLOC_D));
#@HBM_STACK1_ALLOC = split(",",join(",",@HBM_STACK1_ALLOC_D));
#$length = @HBM_STACK0_ALLOC;
#if ( $debug eq "true" ) {print "Post D: Current allocationat stack0($length): @HBM_STACK0_ALLOC\n";}
#$length = @HBM_STACK1_ALLOC;
#if ( $debug eq "true" ) {print "Post D: Current allocationat stack1($length): @HBM_STACK1_ALLOC\n";}

#1.2 allocate Weights
my $ref0;
my $ref1;
my $ref0_ports;
my $ref1_ports;
my @weight_ports = qw(W0 W1);
$current_stage = "W";
given ($FORCE_W_LOC) {
    when ($_ eq "0" ) {
	if ($SLR0 ne "0") { 
	    $ref0 = \@weight_ports;
	    $ref0_ports = expand_port_name("dpu_0",$pprefix,$ref0);
	    push (@HBM_STACK0_ALLOC_W,@$ref0_ports);
	}
	if ($SLR1 ne "0") { 
	    $ref0 = \@weight_ports;
	    $ref0_ports = expand_port_name("dpu_1",$pprefix,$ref0);
	    push (@HBM_STACK0_ALLOC_W,@$ref0_ports);
	}
	if ($SLR2 ne "0") { 
	    $ref0 = \@weight_ports;
	    $ref0_ports = expand_port_name("dpu_2",$pprefix,$ref0);
	    push (@HBM_STACK0_ALLOC_W,@$ref0_ports);
	}
    }
    when ($_ eq "1" ) {
	if ($SLR0 ne "0") { 
	    $ref1 = \@weight_ports;
	    $ref1_ports = expand_port_name("dpu_0",$pprefix,$ref1);
	    push (@HBM_STACK1_ALLOC_W,@$ref1_ports);
	}
	if ($SLR1 ne "0") { 
	    $ref1 = \@weight_ports;
	    $ref1_ports = expand_port_name("dpu_1",$pprefix,$ref1);
	    push (@HBM_STACK1_ALLOC_W,@$ref1_ports);
	}
	if ($SLR2 ne "0") { 
	    $ref1 = \@weight_ports;
	    $ref1_ports = expand_port_name("dpu_2",$pprefix,$ref1);
	    push (@HBM_STACK1_ALLOC_W,@$ref1_ports);
	}
    }
    when ($_ eq "balance" ) {
	my $current_statck0_length = @HBM_STACK0_ALLOC_D;
	my $current_statck1_length = @HBM_STACK1_ALLOC_D;
	#print "debug: @HBM_STACK0_ALLOC_D\n";
	#print "debug: @HBM_STACK1_ALLOC_D\n";

	my $init = $current_statck0_length - $current_statck1_length;
	my $step = 2;
	my $iter_num = @slr_encode;
	#print "Init: $current_statck0_length and $current_statck1_length $init, $step, $iter_num\n";
	$iter = 0;
	@result0 = ();
	@result1 = ();
	allocate($init, $step, $iter_num);
	my $my_index;
	foreach $my_index (@result0) {
	    my $dpu_index = $slr_encode[$my_index];    

	    my $ref = \@weight_ports;
	    my $ref_ports = expand_port_name("$dpu_index",$pprefix,$ref);

	    push (@HBM_STACK0_ALLOC_W,@$ref_ports);
	}
	foreach $my_index (@result1) {
	    my $dpu_index = $slr_encode[$my_index];    

	    my $ref = \@weight_ports;
	    my $ref_ports = expand_port_name("$dpu_index",$pprefix,$ref);

	    push (@HBM_STACK1_ALLOC_W,@$ref_ports);
	}
    }
    default {
	print "ERROR: not recognized options for FORCE_W_LOC - $FORCE_W_LOC\n";
    }
}
#print "aa post W: @result0\n";
#print "bb post W: @result1\n";
if ($debug eq "true" ) {print "Current W assignment at stack0: @HBM_STACK0_ALLOC_W\n";} 
if ($debug eq "true" ) {print "Current W assignment at stack1: @HBM_STACK1_ALLOC_W\n";}
#$length = @HBM_STACK0_ALLOC;
#if ( $debug eq "true" ) {print "Post W: Current allocationat stack0($length): @HBM_STACK0_ALLOC\n";}
#$length = @HBM_STACK1_ALLOC;
#if ( $debug eq "true" ) {print "Post W: Current allocationat stack1($length): @HBM_STACK1_ALLOC\n";}
if ($w_cross_hbm_switch eq "true") {
    #push (@HBM_STACK0_ALLOC, split(",",join(",",@HBM_STACK0_ALLOC_W)));
    #push (@HBM_STACK1_ALLOC, split(",",join(",",@HBM_STACK1_ALLOC_W)));
} else {
    my $w_st0_legnth = @HBM_STACK0_ALLOC_W;
    my $w_st1_legnth = @HBM_STACK1_ALLOC_W;
    my $current_st0_length = @HBM_STACK0_ALLOC_D;
    my $current_st1_length = @HBM_STACK1_ALLOC_D;

    if ( $w_st0_legnth >= 5 or $w_st1_legnth >= 5) {
	print "ERROR: either W ports in stack0 or stack1 need MC more than 4, cross-switch must happen. So ignore w_cross_hbm_switch\n";
    } else {
	my $i=0;
	if (ceil(($w_st0_legnth+$current_st0_length)/4) != ceil($current_st0_length/4)) {
	    print "Info: W ports will cross switch at stack0, so rearrange W's MC\n";
	    if ($current_st0_length%4 != 0) {
	        for ($i=0; $i<4-$current_st0_length%4;$i++) {
	            unshift (@HBM_STACK0_ALLOC_W, "null");
	        }
	    }
	}
	#if ($AXI_SHARE eq "false") { 
        #    push (@HBM_STACK0_ALLOC, split(",",join(",",@HBM_STACK0_ALLOC_W)));
	#}

	if (ceil(($w_st1_legnth+$current_st0_length)/4) != ceil($current_st1_length/4)) {
	    print "Info: W ports will cross switch at stack1, so rearrange W's MC\n";
	    if ($current_st1_length%4 != 0) {
	        for ($i=0; $i<4-$current_st1_length%4;$i++) {
	            unshift (@HBM_STACK1_ALLOC_W, "null");
	        }
	    }
	}
	#if ($AXI_SHARE eq "false") { 
        #    push (@HBM_STACK1_ALLOC, split(",",join(",",@HBM_STACK1_ALLOC_W)));
	#}
    }

}

#If we use AXI_SHARE, W0 W1 are disappear, only I is used
if ($AXI_SHARE eq "true") { 
    @HBM_STACK0_ALLOC_W = ();
    @HBM_STACK1_ALLOC_W = ();
}
#print "@HBM_STACK0_ALLOC\n";
#print "@HBM_STACK1_ALLOC\n";



#1.3 Instruction
my @instr_ports = qw(I0);
$current_stage = "I";
given ($FORCE_I_LOC) {
    when ($_ eq "0" ) {
	if ($SLR0 ne "0") { 
	    $ref0 = \@instr_ports;
	    $ref0_ports = expand_port_name("dpu_0",$pprefix,$ref0);
	    push (@HBM_STACK0_ALLOC_I,@$ref0_ports);
	}
	if ($SLR1 ne "0") { 
	    $ref0 = \@instr_ports;
	    $ref0_ports = expand_port_name("dpu_1",$pprefix,$ref0);
	    push (@HBM_STACK0_ALLOC_I,@$ref0_ports);
	}
	if ($SLR2 ne "0") { 
	    $ref0 = \@instr_ports;
	    $ref0_ports = expand_port_name("dpu_2",$pprefix,$ref0);
	    push (@HBM_STACK0_ALLOC_I,@$ref0_ports);
	}
    }
    when ($_ eq "1" ) {
	if ($SLR0 ne "0") { 
	    $ref1 = \@instr_ports;
	    $ref1_ports = expand_port_name("dpu_0",$pprefix,$ref1);
	    push (@HBM_STACK1_ALLOC_I,@$ref1_ports);
	}
	if ($SLR1 ne "0") { 
	    $ref1 = \@instr_ports;
	    $ref1_ports = expand_port_name("dpu_1",$pprefix,$ref1);
	    push (@HBM_STACK1_ALLOC_I,@$ref1_ports);
	}
	if ($SLR2 ne "0") { 
	    $ref1 = \@instr_ports;
	    $ref1_ports = expand_port_name("dpu_2",$pprefix,$ref1);
	    push (@HBM_STACK1_ALLOC_I,@$ref1_ports);
	}
    }
    when ($_ eq "balance" ) {
	my $current_statck0_D_length = @HBM_STACK0_ALLOC_D;
	my $current_statck1_D_length = @HBM_STACK1_ALLOC_D;
	my $current_statck0_W_length = @HBM_STACK0_ALLOC_W;
	my $current_statck1_W_length = @HBM_STACK1_ALLOC_W;
	my $current_statck0_length = $current_statck0_D_length + $current_statck0_W_length;
	my $current_statck1_length = $current_statck1_D_length + $current_statck1_W_length;
	#print "debug: @HBM_STACK0_ALLOC\n";
	#print "debug: @HBM_STACK1_ALLOC\n";

	my $init = $current_statck0_length - $current_statck1_length;
	my $step = 1;
	my $iter_num = @slr_encode;
	#print "Init: $current_statck0_length and $current_statck1_length $init, $step, $iter_num\n";
	$iter = 0;
	@result0 = ();
	@result1 = ();
	allocate($init, $step, $iter_num);
	my $my_index;
	foreach $my_index (@result0) {
	    my $dpu_index = $slr_encode[$my_index];    

	    my $ref = \@instr_ports;
	    my $ref_ports = expand_port_name("$dpu_index",$pprefix,$ref);

	    push (@HBM_STACK0_ALLOC_I,@$ref_ports);
	}
	foreach $my_index (@result1) {
	    my $dpu_index = $slr_encode[$my_index];    

	    my $ref = \@instr_ports;
	    my $ref_ports = expand_port_name("$dpu_index",$pprefix,$ref);

	    push (@HBM_STACK1_ALLOC_I,@$ref_ports);
	}
    }
    default {
	print "ERROR: not recognized options for FORCE_I_LOC - $FORCE_I_LOC\n";
    }
}

#print "aa post I: @result0\n";
#print "bb post I: @result1\n";
if ( $debug eq "true" ) {print "Post I assignment at stack0: @HBM_STACK0_ALLOC_I\n";}
if ( $debug eq "true" ) {print "Post I assignment at stack1: @HBM_STACK1_ALLOC_I\n";}
#push (@HBM_STACK0_ALLOC, split(",",join(",",@HBM_STACK0_ALLOC_I)));
#push (@HBM_STACK1_ALLOC, split(",",join(",",@HBM_STACK1_ALLOC_I)));
#$length = @HBM_STACK0_ALLOC;
#if ( $debug eq "true" ) {print "Post I: Current allocationat stack0($length): @HBM_STACK0_ALLOC\n";}
#$length = @HBM_STACK1_ALLOC;
#if ( $debug eq "true" ) {print "Post I: Current allocationat stack1($length): @HBM_STACK1_ALLOC\n";}

#Now, locate DWI into stacks
given ($FORCE_D_LOC) {
    when ($_ eq "balance") {
	#stack0
	@HBM_STACK0_ALLOC = split(",",join(",",@HBM_LEFT_ALLOC_D));
        push (@HBM_STACK0_ALLOC, split(",",join(",",@HBM_STACK0_ALLOC_W)));
        push (@HBM_STACK0_ALLOC, split(",",join(",",@HBM_STACK0_ALLOC_I)));
	#stack1
	@HBM_STACK1_ALLOC = split(",",join(",",@HBM_RIGHT_ALLOC_D));
        push (@HBM_STACK1_ALLOC, split(",",join(",",@HBM_STACK1_ALLOC_W)));
        push (@HBM_STACK1_ALLOC, split(",",join(",",@HBM_STACK1_ALLOC_I)));
	
    }
    when ($_ eq "0") {
	#stack0
	@HBM_STACK0_ALLOC = split(",",join(",",@HBM_LEFT_ALLOC_D));
        push (@HBM_STACK0_ALLOC, split(",",join(",",@HBM_STACK0_ALLOC_W)));
        push (@HBM_STACK0_ALLOC, split(",",join(",",@HBM_STACK0_ALLOC_I)));
        push (@HBM_STACK0_ALLOC, split(",",join(",",@HBM_RIGHT_ALLOC_D)));
	#stack1
	@HBM_STACK1_ALLOC = split(",",join(",",@HBM_STACK1_ALLOC_W));
        push (@HBM_STACK1_ALLOC, split(",",join(",",@HBM_STACK1_ALLOC_I)));
    }
    when ($_ eq "1") {
	#stack0
	@HBM_STACK0_ALLOC = split(",",join(",",@HBM_STACK0_ALLOC_W));
        push (@HBM_STACK0_ALLOC, split(",",join(",",@HBM_STACK0_ALLOC_I)));
	#stack1
	@HBM_STACK1_ALLOC = split(",",join(",",@HBM_LEFT_ALLOC_D));
        push (@HBM_STACK1_ALLOC, split(",",join(",",@HBM_STACK1_ALLOC_W)));
        push (@HBM_STACK1_ALLOC, split(",",join(",",@HBM_STACK1_ALLOC_I)));
        push (@HBM_STACK1_ALLOC, split(",",join(",",@HBM_RIGHT_ALLOC_D)));
    }
    default {
	print "ERROR: not recognized options for FORCE_D_LOC - $FORCE_D_LOC\n";
    }
}


#Manually HBM MC allocation
if ($MANUALLY_HBM_ASSIGN ne "" ) {
    print "Info: use user's manual HBM MC assignment file\n";
    my $i;	
    open ( M, "$MANUALLY_HBM_ASSIGN") or die "Failed to Open File $MANUALLY_HBM_ASSIGN, $!\n";
    @HBM_STACK0_ALLOC = ();
    @HBM_STACK1_ALLOC = ();
    for ($i=0; $i<16;$i++) {
	$HBM_STACK0_ALLOC[$i] = "null";
	$HBM_STACK1_ALLOC[$i] = "null";
    }
    while (<M>) {
	if ($_ =~ /^\s*(\S+)\s+(\d+)/) {
	    my $port = $1;
	    my $index = $2;
	    if ($index < 16) {
		$HBM_STACK0_ALLOC[$index] = $port;
		#print "aa $index @HBM_STACK0_ALLOC\n";
	    } else {
		my $index = $index - 16;
		$HBM_STACK1_ALLOC[$index] = $port;
		#print "bb $index @HBM_STACK1_ALLOC\n";
	    }
	}
    }
    close (M);
}

#print "@HBM_STACK0_ALLOC\n";
#print "@HBM_STACK1_ALLOC\n";

#===================================================================================================
#output
my @locs;
#1. for Vitis
open ( TCL , "> $SYS_LINK_POST" ) or die "Failed to Open File $SYS_LINK_POST, $!\n" ;
open ( T, "> $TXT" ) or die "Failed to Open File $TXT, $!\n" ;
open ( HBM_RS_PBLOCK , "> ./bit_gen/constraints/HBM_RS_pblock.xdc" ) or die "Failed to Open File ./bit_gen/constraints/HBM_RS_pblock.xdc, $!\n" ;
open ( HBM_RS_INSERT , "> ./bit_gen/constraints/HBM_RS_insert.tcl" ) or die "Failed to Open File ./bit_gen/constraints/HBM_RS_insert.tcl, $!\n" ;

if ( $ARCH eq "v3e" ) { 
    if ( $SLR2 ne 0 ) {
        print HBM_RS_PBLOCK "create_pblock pblock_dynamic_HBM_RS_S\n" ;
        print HBM_RS_PBLOCK "create_pblock pblock_dynamic_HBM_RS_M\n" ;
        print HBM_RS_PBLOCK "resize_pblock [get_pblocks pblock_dynamic_HBM_RS_S] -add {CLOCKREGION_X0Y4:CLOCKREGION_X6Y5}\n" ;
        print HBM_RS_PBLOCK "resize_pblock [get_pblocks pblock_dynamic_HBM_RS_M] -add {CLOCKREGION_X0Y2:CLOCKREGION_X6Y3}\n" ;
        print HBM_RS_PBLOCK "set_property PARENT pblock_dynamic_SLR0 [get_pblocks pblock_dynamic_HBM_RS_M]\n" ;
        print HBM_RS_PBLOCK "set_property PARENT pblock_dynamic_SLR1 [get_pblocks pblock_dynamic_HBM_RS_S]\n" ;
        print HBM_RS_PBLOCK "\n" x 3 ;
    }
}

my $host_hbm_assignment = "set_property -dict [list CONFIG.S00_MEM { \\\n";
my $hbm_index;

#compute host ports
my $s0_length = @HBM_STACK0_ALLOC;
my $s1_length = @HBM_STACK1_ALLOC;
if ($s1_length == 0 ) {#all ports locate at stack0
	print "Debug 1 $s1_length\n";
    #$host_port_alloc = $s0_length;
} elsif ($s0_length == 0 ) {#all ports locate at stack1
	print "Debug 2 $s1_length\n";
    #$host_port_alloc = $s1_length;
} else {#ports locate at both stack0 and stack1
    #$host_port_alloc = 8
}
print TCL "hbm_memory_subsystem::force_host_port $host_port_alloc 1 [get_bd_cells hmss_0]\n";

$length = @HBM_STACK0_ALLOC;
for ($iter=0; $iter<$length; $iter++) {
    #ingore null ports
    if ($HBM_STACK0_ALLOC[$iter] eq "null") {
	next;
    }

    #dealing function ports
    my $loc = $iter;
    push (@locs, $loc);
    print TCL "hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins $HBM_STACK0_ALLOC[$iter]] $loc 1 [get_bd_cells hmss_0]\n";
    print T "$HBM_STACK0_ALLOC[$iter] $loc\n";
    $hbm_index = sprintf "%02s", $loc;
    $host_hbm_assignment .= "HBM_MEM$hbm_index \\\n";

    if ( $ARCH eq "v3e" ) { 
        if( $HBM_STACK0_ALLOC[$iter] =~ /dpu_2/ ) {
            print HBM_RS_PBLOCK "if {[llength [get_cells \"$shell_hier/hmss_0/inst/path_${loc}/interconnect*/inst\"]] > 0} {\n" ;
            print HBM_RS_PBLOCK "\tadd_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_S] [get_cells \"$shell_hier/hmss_0/inst/path_${loc}/interconnect*/inst/s00_entry_pipeline\"]\n" ;
            print HBM_RS_PBLOCK "\tadd_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_M] [get_cells \"$shell_hier/hmss_0/inst/path_${loc}/interconnect*/inst/m00_exit_pipeline\"]\n" ;
           #print HBM_RS_PBLOCK "} elseif {[llength [get_cells \"$shell_hier/hmss_0/inst/path_${loc}/switch*/inst\"]] > 0} {\n" ;
           #print HBM_RS_PBLOCK "\tadd_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_S] [get_cells \"$shell_hier/hmss_0/inst/path_${loc}/switch*/inst/s00_entry_pipeline\"]\n" ;
           #print HBM_RS_PBLOCK "\tadd_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_M] [get_cells \"$shell_hier/hmss_0/inst/path_${loc}/switch*/inst/m00_exit_pipeline\"]\n" ;
            print HBM_RS_PBLOCK "} else { \n" ;
            print HBM_RS_PBLOCK "\tsend_msg_id {HBM_RS_PBLOCK 1-2} INFO \"$shell_hier/hmss_0/inst/path_${loc} neither interconnect nor switch found !\"\n" ;
            print HBM_RS_PBLOCK "} \n" ;

            my $ID = trim_DPU_AXI ("$HBM_STACK0_ALLOC[$iter]") ;
            print HBM_RS_INSERT "set one [get_bd_intf_pins /dpu_2/DPU_AXI_$ID]\n" ;
            print HBM_RS_INSERT "set another [get_bd_intf_pins -of [get_bd_intf_nets -of \$one] -filter {MODE==Slave}]\n" ;
            print HBM_RS_INSERT "delete_bd_objs [get_bd_intf_nets -of [get_bd_intf_pins /dpu_2/DPU_AXI_$ID]]\n" ;
            print HBM_RS_INSERT "create_bd_cell -type ip -vlnv xilinx.com:ip:axi_register_slice:2.1 axi_register_slice_$ID\n" ;
            print HBM_RS_INSERT "connect_bd_intf_net \$one [get_bd_intf_pins axi_register_slice_$ID/S_AXI]\n" ;
            print HBM_RS_INSERT "connect_bd_intf_net [get_bd_intf_pins axi_register_slice_$ID/M_AXI] \$another\n" ;
            print HBM_RS_INSERT "connect_bd_net [get_bd_pins /dpu_2/ap_clk] [get_bd_pins axi_register_slice_$ID/aclk]\n" ;
            print HBM_RS_INSERT "connect_bd_net [get_bd_pins axi_register_slice_$ID/aresetn] [get_bd_pins /dpu_2/ap_rst_n]\n" ;
			if ($ALVEO eq "u55c") {
            print HBM_RS_INSERT "set_property -dict [list CONFIG.PROTOCOL.VALUE_SRC USER CONFIG.ID_WIDTH.VALUE_SRC USER CONFIG.DATA_WIDTH.VALUE_SRC USER CONFIG.ADDR_WIDTH.VALUE_SRC USER CONFIG.READ_WRITE_MODE.VALUE_SRC USER] [get_bd_cells axi_register_slice_$ID]\n" ;
            print HBM_RS_INSERT "set_property -dict [list CONFIG.PROTOCOL {AXI3} CONFIG.ADDR_WIDTH {34} CONFIG.DATA_WIDTH {256} CONFIG.ID_WIDTH {6} CONFIG.REG_AW {15} CONFIG.REG_AR {15} CONFIG.REG_W {15} CONFIG.REG_R {15} CONFIG.REG_B {15} CONFIG.NUM_SLR_CROSSINGS {2} CONFIG.PIPELINES_MASTER_AW {4} CONFIG.PIPELINES_SLAVE_AW {4} CONFIG.PIPELINES_MIDDLE_AW {4} CONFIG.PIPELINES_MASTER_AR {4} CONFIG.PIPELINES_SLAVE_AR {4} CONFIG.PIPELINES_MIDDLE_AR {4} CONFIG.PIPELINES_MASTER_W {4} CONFIG.PIPELINES_SLAVE_W {4} CONFIG.PIPELINES_MIDDLE_W {4} CONFIG.PIPELINES_MASTER_R {4} CONFIG.PIPELINES_SLAVE_R {4} CONFIG.PIPELINES_MIDDLE_R {4} CONFIG.PIPELINES_MASTER_B {4} CONFIG.PIPELINES_SLAVE_B {4} CONFIG.PIPELINES_MIDDLE_B {4}] [get_bd_cells axi_register_slice_$ID]\n" ;
			}
        }
        elsif( $HBM_STACK0_ALLOC[$iter] =~ /dpu_1/ ) {
            my $ID = trim_DPU_AXI ("$HBM_STACK0_ALLOC[$iter]") ;
            print HBM_RS_INSERT "set one [get_bd_intf_pins /dpu_1/DPU_AXI_$ID]\n" ;
            print HBM_RS_INSERT "set another [get_bd_intf_pins -of [get_bd_intf_nets -of \$one] -filter {MODE==Slave}]\n" ;
            print HBM_RS_INSERT "delete_bd_objs [get_bd_intf_nets -of [get_bd_intf_pins /dpu_1/DPU_AXI_$ID]]\n" ;
            print HBM_RS_INSERT "create_bd_cell -type ip -vlnv xilinx.com:ip:axi_register_slice:2.1 axi_register_slice_dpu_1_$ID\n" ;
            print HBM_RS_INSERT "connect_bd_intf_net \$one [get_bd_intf_pins axi_register_slice_dpu_1_$ID/S_AXI]\n" ;
            print HBM_RS_INSERT "connect_bd_intf_net [get_bd_intf_pins axi_register_slice_dpu_1_$ID/M_AXI] \$another\n" ;
            print HBM_RS_INSERT "connect_bd_net [get_bd_pins /dpu_1/ap_clk] [get_bd_pins axi_register_slice_dpu_1_$ID/aclk]\n" ;
            print HBM_RS_INSERT "connect_bd_net [get_bd_pins axi_register_slice_dpu_1_$ID/aresetn] [get_bd_pins /dpu_1/ap_rst_n]\n" ;
			if ($ALVEO eq "u55c") {
            print HBM_RS_INSERT "set_property -dict [list CONFIG.PROTOCOL.VALUE_SRC USER CONFIG.ID_WIDTH.VALUE_SRC USER CONFIG.DATA_WIDTH.VALUE_SRC USER CONFIG.ADDR_WIDTH.VALUE_SRC USER CONFIG.READ_WRITE_MODE.VALUE_SRC USER] [get_bd_cells axi_register_slice_dpu_1_$ID]\n" ;
            print HBM_RS_INSERT "set_property -dict [list CONFIG.PROTOCOL {AXI3} CONFIG.ADDR_WIDTH {34} CONFIG.DATA_WIDTH {256} CONFIG.ID_WIDTH {6} CONFIG.REG_AW {15} CONFIG.REG_AR {15} CONFIG.REG_W {15} CONFIG.REG_R {15} CONFIG.REG_B {15} CONFIG.NUM_SLR_CROSSINGS {2} CONFIG.PIPELINES_MASTER_AW {4} CONFIG.PIPELINES_SLAVE_AW {4} CONFIG.PIPELINES_MIDDLE_AW {4} CONFIG.PIPELINES_MASTER_AR {4} CONFIG.PIPELINES_SLAVE_AR {4} CONFIG.PIPELINES_MIDDLE_AR {4} CONFIG.PIPELINES_MASTER_W {4} CONFIG.PIPELINES_SLAVE_W {4} CONFIG.PIPELINES_MIDDLE_W {4} CONFIG.PIPELINES_MASTER_R {4} CONFIG.PIPELINES_SLAVE_R {4} CONFIG.PIPELINES_MIDDLE_R {4} CONFIG.PIPELINES_MASTER_B {4} CONFIG.PIPELINES_SLAVE_B {4} CONFIG.PIPELINES_MIDDLE_B {4}] [get_bd_cells axi_register_slice_dpu_1_$ID]\n" ;
			}
        }
    }
}
$length = @HBM_STACK1_ALLOC;
for ($iter=0; $iter<$length; $iter++) {
    #ingore null ports
    if ($HBM_STACK1_ALLOC[$iter] eq "null") {
	next;
    }

    #dealing function ports
    my $loc = 16 + $iter;
    push (@locs, $loc);
    print TCL "hbm_memory_subsystem::force_kernel_port [get_bd_intf_pins $HBM_STACK1_ALLOC[$iter]] $loc 1 [get_bd_cells hmss_0]\n";
    print T "$HBM_STACK1_ALLOC[$iter] $loc\n";
    $hbm_index = sprintf "%02s", $loc;
    $host_hbm_assignment .= "HBM_MEM$hbm_index \\\n";

    if ( $ARCH eq "v3e" ) { 
        if( $HBM_STACK1_ALLOC[$iter] =~ /dpu_2/ ) {
            print HBM_RS_PBLOCK "if {[llength [get_cells \"$shell_hier/hmss_0/inst/path_${loc}/interconnect*/inst\"]] > 0} {\n" ;
            print HBM_RS_PBLOCK "\tadd_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_S] [get_cells \"$shell_hier/hmss_0/inst/path_${loc}/interconnect*/inst/s00_entry_pipeline\"]\n" ;
            print HBM_RS_PBLOCK "\tadd_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_M] [get_cells \"$shell_hier/hmss_0/inst/path_${loc}/interconnect*/inst/m00_exit_pipeline\"]\n" ;
           #print HBM_RS_PBLOCK "} elseif {[llength [get_cells \"$shell_hier/hmss_0/inst/path_${loc}/switch*/inst\"]] > 0} {\n" ;
           #print HBM_RS_PBLOCK "\tadd_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_S] [get_cells \"$shell_hier/hmss_0/inst/path_${loc}/switch*/inst/s00_entry_pipeline\"]\n" ;
           #print HBM_RS_PBLOCK "\tadd_cells_to_pblock [get_pblocks pblock_dynamic_HBM_RS_M] [get_cells \"$shell_hier/hmss_0/inst/path_${loc}/switch*/inst/m00_exit_pipeline\"]\n" ;
            print HBM_RS_PBLOCK "} else { \n" ;
            print HBM_RS_PBLOCK "\tsend_msg_id {HBM_RS_PBLOCK 1-2} INFO \"$shell_hier/hmss_0/inst/path_${loc} neither interconnect nor switch found !\"\n" ;
            print HBM_RS_PBLOCK "} \n" ;

            my $ID = trim_DPU_AXI ("$HBM_STACK1_ALLOC[$iter]") ;

            print HBM_RS_INSERT "set one [get_bd_intf_pins /dpu_2/DPU_AXI_$ID]\n" ;
            print HBM_RS_INSERT "set another [get_bd_intf_pins -of [get_bd_intf_nets -of \$one] -filter {MODE==Slave}]\n" ;
            print HBM_RS_INSERT "delete_bd_objs [get_bd_intf_nets -of [get_bd_intf_pins /dpu_2/DPU_AXI_$ID]]\n" ;
            print HBM_RS_INSERT "create_bd_cell -type ip -vlnv xilinx.com:ip:axi_register_slice:2.1 axi_register_slice_$ID\n" ;
            print HBM_RS_INSERT "connect_bd_intf_net \$one [get_bd_intf_pins axi_register_slice_$ID/S_AXI]\n" ;
            print HBM_RS_INSERT "connect_bd_intf_net [get_bd_intf_pins axi_register_slice_$ID/M_AXI] \$another\n" ;
            print HBM_RS_INSERT "connect_bd_net [get_bd_pins /dpu_2/ap_clk] [get_bd_pins axi_register_slice_$ID/aclk]\n" ;
            print HBM_RS_INSERT "connect_bd_net [get_bd_pins axi_register_slice_$ID/aresetn] [get_bd_pins /dpu_2/ap_rst_n]\n" ;
			if ($ALVEO eq "u55c") {
            print HBM_RS_INSERT "set_property -dict [list CONFIG.PROTOCOL.VALUE_SRC USER CONFIG.ID_WIDTH.VALUE_SRC USER CONFIG.DATA_WIDTH.VALUE_SRC USER CONFIG.ADDR_WIDTH.VALUE_SRC USER CONFIG.READ_WRITE_MODE.VALUE_SRC USER] [get_bd_cells axi_register_slice_$ID]\n" ;
            print HBM_RS_INSERT "set_property -dict [list CONFIG.PROTOCOL {AXI3} CONFIG.ADDR_WIDTH {34} CONFIG.DATA_WIDTH {256} CONFIG.ID_WIDTH {6} CONFIG.REG_AW {15} CONFIG.REG_AR {15} CONFIG.REG_W {15} CONFIG.REG_R {15} CONFIG.REG_B {15} CONFIG.NUM_SLR_CROSSINGS {2} CONFIG.PIPELINES_MASTER_AW {4} CONFIG.PIPELINES_SLAVE_AW {4} CONFIG.PIPELINES_MIDDLE_AW {4} CONFIG.PIPELINES_MASTER_AR {4} CONFIG.PIPELINES_SLAVE_AR {4} CONFIG.PIPELINES_MIDDLE_AR {4} CONFIG.PIPELINES_MASTER_W {4} CONFIG.PIPELINES_SLAVE_W {4} CONFIG.PIPELINES_MIDDLE_W {4} CONFIG.PIPELINES_MASTER_R {4} CONFIG.PIPELINES_SLAVE_R {4} CONFIG.PIPELINES_MIDDLE_R {4} CONFIG.PIPELINES_MASTER_B {4} CONFIG.PIPELINES_SLAVE_B {4} CONFIG.PIPELINES_MIDDLE_B {4}] [get_bd_cells axi_register_slice_$ID]\n" ;
			}
        }
        elsif( $HBM_STACK1_ALLOC[$iter] =~ /dpu_1/ ) {
			if ($ALVEO eq "u55c") {
            my $ID = trim_DPU_AXI ("$HBM_STACK1_ALLOC[$iter]") ;
            print HBM_RS_INSERT "set one [get_bd_intf_pins /dpu_1/DPU_AXI_$ID]\n" ;
            print HBM_RS_INSERT "set another [get_bd_intf_pins -of [get_bd_intf_nets -of \$one] -filter {MODE==Slave}]\n" ;
            print HBM_RS_INSERT "delete_bd_objs [get_bd_intf_nets -of [get_bd_intf_pins /dpu_1/DPU_AXI_$ID]]\n" ;
            print HBM_RS_INSERT "create_bd_cell -type ip -vlnv xilinx.com:ip:axi_register_slice:2.1 axi_register_slice_dpu_1_$ID\n" ;
            print HBM_RS_INSERT "connect_bd_intf_net \$one [get_bd_intf_pins axi_register_slice_dpu_1_$ID/S_AXI]\n" ;
            print HBM_RS_INSERT "connect_bd_intf_net [get_bd_intf_pins axi_register_slice_dpu_1_$ID/M_AXI] \$another\n" ;
            print HBM_RS_INSERT "connect_bd_net [get_bd_pins /dpu_1/ap_clk] [get_bd_pins axi_register_slice_dpu_1_$ID/aclk]\n" ;
            print HBM_RS_INSERT "connect_bd_net [get_bd_pins axi_register_slice_dpu_1_$ID/aresetn] [get_bd_pins /dpu_1/ap_rst_n]\n" ;
            print HBM_RS_INSERT "set_property -dict [list CONFIG.PROTOCOL.VALUE_SRC USER CONFIG.ID_WIDTH.VALUE_SRC USER CONFIG.DATA_WIDTH.VALUE_SRC USER CONFIG.ADDR_WIDTH.VALUE_SRC USER CONFIG.READ_WRITE_MODE.VALUE_SRC USER] [get_bd_cells axi_register_slice_dpu_1_$ID]\n" ;
            print HBM_RS_INSERT "set_property -dict [list CONFIG.PROTOCOL {AXI3} CONFIG.ADDR_WIDTH {34} CONFIG.DATA_WIDTH {256} CONFIG.ID_WIDTH {6} CONFIG.REG_AW {15} CONFIG.REG_AR {15} CONFIG.REG_W {15} CONFIG.REG_R {15} CONFIG.REG_B {15} CONFIG.NUM_SLR_CROSSINGS {2} CONFIG.PIPELINES_MASTER_AW {4} CONFIG.PIPELINES_SLAVE_AW {4} CONFIG.PIPELINES_MIDDLE_AW {4} CONFIG.PIPELINES_MASTER_AR {4} CONFIG.PIPELINES_SLAVE_AR {4} CONFIG.PIPELINES_MIDDLE_AR {4} CONFIG.PIPELINES_MASTER_W {4} CONFIG.PIPELINES_SLAVE_W {4} CONFIG.PIPELINES_MIDDLE_W {4} CONFIG.PIPELINES_MASTER_R {4} CONFIG.PIPELINES_SLAVE_R {4} CONFIG.PIPELINES_MIDDLE_R {4} CONFIG.PIPELINES_MASTER_B {4} CONFIG.PIPELINES_SLAVE_B {4} CONFIG.PIPELINES_MIDDLE_B {4}] [get_bd_cells axi_register_slice_dpu_1_$ID]\n" ;
			}

       }
    }
}

$host_hbm_assignment .= "}] [get_bd_cells hmss_0]\n";
if ($FORCE_ALL_HBM_ASSIGN eq "true") {
    $host_hbm_assignment = "";
}
print TCL "\n$host_hbm_assignment";
#2. for vivado's implementation xdc
#my $xdc = "./impl.xdc";
#open ( XDC, "> $xdc" ) or die "Failed to Open File $xdc, $!\n" ;
#foreach (@locs) {
#    print XDC "pfm_top_i/L1/L1_URP/hmss_0/inst/path_$_/interconnect*/inst/s00_entry_pipeline \\\n";
#}
#close (XDC);

if ( $ARCH eq "v3e" ) { 
    if ( $SLR2 ne 0 ) {
        printf HBM_RS_INSERT "\n" x 3 ;
       #printf HBM_RS_INSERT "set_property -dict [list CONFIG.S00_SLR {SLR2}] [get_bd_cells hmss_0]\n" ;
        printf HBM_RS_INSERT "assign_bd_address [get_bd_addr_segs {dpu_2/s_axi_control/reg0}]\n" ;
    }
}
close (HBM_RS_PBLOCK);


print TCL "\n" ;

if ($ALVEO eq "u280") {
    open ( CLEANUP, '<:encoding(UTF-8)', "./bit_gen/constraints/sys_link_post_cleanup_Mike.${DEVICE}.tcl" ) or die "Failed to Open File sys_link_post_cleanup_Mike.${DEVICE}.tcl, $! \n" ;
    while ( my $row = <CLEANUP> ) {
        print TCL "$row" ;
    }
    open ( HBM_RS_INSERT, '<:encoding(UTF-8)', "./bit_gen/constraints/HBM_RS_insert.tcl" ) or die "Failed to Open File HBM_RS_insert.tcl, $! \n" ;
    while ( my $row = <HBM_RS_INSERT> ) {
        print TCL "$row" ;
    }
    print TCL "\n" ;
    print TCL "set_param bd.hooks.addr.debug_scoped_use_ms_name true\n" ;
    print TCL "validate_bd_design -force\n" ;
} elsif ($ALVEO eq "u55c") {
    open ( CLEANUP, '<:encoding(UTF-8)', "./bit_gen/constraints/sys_link_post_cleanup_Mike.${DEVICE}.tcl" ) or die "Failed to Open File sys_link_post_cleanup_Mike.${DEVICE}.tcl, $! \n" ;
    while ( my $row = <CLEANUP> ) {
        print TCL "$row" ;
    }
    open ( HBM_RS_INSERT, '<:encoding(UTF-8)', "./bit_gen/constraints/HBM_RS_insert.tcl" ) or die "Failed to Open File HBM_RS_insert.tcl, $! \n" ;
    while ( my $row = <HBM_RS_INSERT> ) {
        print TCL "$row" ;
    }
    print TCL "\n" ;
    print TCL "set_param bd.hooks.addr.debug_scoped_use_ms_name true\n" ;
    print TCL "validate_bd_design -force\n" ;
} elsif ($ALVEO eq "u50") {
    if ( $u50_rmvmss eq "false") {
        print TCL "\n" ;
        #print TCL "set ap [get_property CONFIG.ADVANCED_PROPERTIES [get_bd_cells /memory_subsystem]]\n" ;
        #print TCL "dict set ap minimal_initial_configuration true\n" ;
        #print TCL "set_property CONFIG.ADVANCED_PROPERTIES \$ap [get_bd_cells /memory_subsystem]\n" ;
    } elsif ( $u50_rmvmss eq "true") {
        open ( CLEANUP, '<:encoding(UTF-8)', "./bit_gen/constraints/sys_link_post_cleanup_Mike.u50.tcl" ) or die "Failed to Open File sys_link_post_cleanup_Mike.u50.tcl, $! \n" ;
        while ( my $row = <CLEANUP> ) {
            print TCL "$row" ;
        }
        #open ( HBM_RS_INSERT, '<:encoding(UTF-8)', "./bit_gen/constraints/HBM_RS_insert.tcl" ) or die "Failed to Open File HBM_RS_insert.tcl, $! \n" ;
        #while ( my $row = <HBM_RS_INSERT> ) {
        #    print TCL "$row" ;
        #}
    } else {
	print "ERROR: Not sure if you want to remove mss on u50\n";
    }

    #decrease HBM's frequency or other setting
    print TCL "\n" ;
    print TCL "set __props [get_property CONFIG.ADVANCED_PROPERTIES [get_bd_cells hmss_0]]\n";
    print TCL "# Put any other IP_OVERRIDE in here!!\n";
    print TCL "dict set __props IP_OVERRIDE hbm_inst {CONFIG.USER_HBM_TCK_0 $HBM_FREQ CONFIG.USER_HBM_TCK_1 $HBM_FREQ}\n";
    print TCL "#shit!\n";
    print TCL "set_property -dict [list CONFIG.ADVANCED_PROPERTIES \$__props] [get_bd_cells hmss_0]\n";

    print TCL "\n" ;
    print TCL "set_param bd.hooks.addr.debug_scoped_use_ms_name true\n" ;
    print TCL "validate_bd_design -force\n" ;

}

close (TCL);
close (T);
close (CLEANUP);

