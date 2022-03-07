#! /usr/bin/env perl -w
use strict ; 
use warnings ;
use 5.010 ;
use File::Basename ;
use Cwd qw ( abs_path ) ;

#my $ABS_DIR = dirname ( abs_path ($0) ) ;
my $ABS_DIR = "." ;

# #############################################################
# Arguments: KRNL STRATEGY FREQ0 FREQ1 SLR0=? SLR1=? SLR2=? \n" ;
# KRNL     : Kernel Name
# STRATEGY : Default, Explore
# FREQ0    :
# FREQ1    :
# SLR0     : 3, 4, 5, 6
# SLR1     : 3, 4, 5, 6
# SLR2     : 3, 4, 5, 6
#
my $KRNL     ;
my $STRATEGY ;
my $FREQ0    ;
my $FREQ1    ;
my $SLR0 = 0 ;
my $SLR1 = 0 ;
my $SLR2 = 0 ;
my $REMOTE_IP_CACHE ;
my $DPU_V3E ;
my $ALVEO ;
my $HBM_PORT_ALLOCATION ;
my $USE_2LV = "false";
my $FORCE_ALL_HBM_ASSIGN = "false";
my $OPT_DESIGN_DIRECTIVE = "Default";
my $PLACE_DIRECTIVE = "Default";
my $XO_PATH = "";

while ( scalar @ARGV > 0 ) {
    my $arg = shift @ARGV ;
    given ( $arg ) {
        when ( /KRNL=([a-zA-Z0-9]{1,})/ ) { $KRNL = $1 ; }
        when ( /STRATEGY=([a-zA-Z]{1,})/ ) { $STRATEGY = $1 ; }
        when ( /FREQ0=([0-9]{1,})/ ) { $FREQ0 = $1 ; }
        when ( /FREQ1=([0-9]{1,})/ ) { $FREQ1 = $1 ; }
        when ( /SLR0=([0-9]{1,})/ ) { $SLR0 = $1 ; }
        when ( /SLR1=([0-9]{1,})/ ) { $SLR1 = $1 ; }
        when ( /SLR2=([0-9]{1,})/ ) { $SLR2 = $1 ; }
        when ( /remote_ip_cache=(.*)/ ) { $REMOTE_IP_CACHE = $1 ; }
        when ( /DPU_V3E=(.*)/ ) { $DPU_V3E = $1 ; }
        when ( /ALVEO=(.*)/ ) { $ALVEO = $1 ; }
        when ( /HBM_PORT_ALLOCATION=(.*)/ ) { $HBM_PORT_ALLOCATION = $1 ; }
        when ( /USE_2LV=(.*)/ ) { $USE_2LV= $1 ; }
        when ( /FORCE_ALL_HBM_ASSIGN=(.*)/ ) { $FORCE_ALL_HBM_ASSIGN = $1 ; }
        when ( /OPT_DESIGN_DIRECTIVE=(.*)/ ) { $OPT_DESIGN_DIRECTIVE = $1 ; }
        when ( /PLACE_DIRECTIVE=(.*)/ ) { $PLACE_DIRECTIVE = $1 ; }
        when ( /XO_PATH=(.*)/ ) { $XO_PATH = $1 ; }
    }
}

my $OF = "./bit_gen/constraints/cons.ini" ;
open ( my $OFH, '>:encoding(UTF-8)', $OF ) 
    or die "Failed to Open File $OF, $!\n" ;

sub advanced {
    print $OFH "\n[advanced]\n" ;
    print $OFH "param=compiler.worstNegativeSlack=-1\n" ;
    print $OFH "param=compiler.errorOnHoldViolation=false\n" ;
    print $OFH "param=compiler.userPostSysLinkTcl=$ABS_DIR/constraints/sys_link_post.tcl\n" ;
    print $OFH "misc=solution_name=link\n" ;
}

sub vivado_default {
    print $OFH "param=project.writeIntermediateCheckpoints=1\n" ;
    print $OFH "prop=run.impl_1.STEPS.OPT_DESIGN.ARGS.DIRECTIVE=$OPT_DESIGN_DIRECTIVE\n" ;
    print $OFH "prop=run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=$PLACE_DIRECTIVE\n" ;
    print $OFH "prop=run.impl_1.STEPS.INIT_DESIGN.TCL.POST=$ABS_DIR/constraints/init_design.post.tcl\n" ;
    print $OFH "prop=run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=$ABS_DIR/constraints/opt_design.pre.tcl\n" ;
    print $OFH "prop=run.impl_1.STEPS.OPT_DESIGN.TCL.POST=$ABS_DIR/constraints/opt_design.post.tcl\n" ;
    print $OFH "prop=run.impl_1.STEPS.PLACE_DESIGN.TCL.POST=$ABS_DIR/constraints/place_design.post.tcl\n" ;
    print $OFH "prop=run.impl_1.{STEPS.PLACE_DESIGN.ARGS.MORE OPTIONS}={-verbose -debug_log}\n" ;
    print $OFH "prop=run.impl_1.STEPS.ROUTE_DESIGN.TCL.POST=$ABS_DIR/constraints/route_design.post.tcl\n" ;
    print $OFH "prop=run.impl_1.{STEPS.ROUTE_DESIGN.ARGS.MORE OPTIONS}={-verbose}\n" ;
    if ( $USE_2LV eq "true" ) {
        print $OFH "prop=run.impl_1.part=xcu50-fsvh2104-2LV-e\n";
    }
}
sub vivado_explore {
    print $OFH "param=project.writeIntermediateCheckpoints=1\n" ;
    print $OFH "prop=run.impl_1.STEPS.OPT_DESIGN.ARGS.DIRECTIVE=$OPT_DESIGN_DIRECTIVE\n" ;
    print $OFH "prop=run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=$PLACE_DIRECTIVE\n" ;
    print $OFH "prop=run.impl_1.STEPS.INIT_DESIGN.TCL.POST=$ABS_DIR/constraints/init_design.post.tcl\n" ;
    print $OFH "prop=run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=$ABS_DIR/constraints/opt_design.pre.tcl\n" ;
    print $OFH "prop=run.impl_1.STEPS.OPT_DESIGN.TCL.POST=$ABS_DIR/constraints/opt_design.post.tcl\n" ;
    print $OFH "prop=run.impl_1.STEPS.PLACE_DESIGN.TCL.POST=$ABS_DIR/constraints/place_design.post.tcl\n" ;
    print $OFH "prop=run.impl_1.{STEPS.PLACE_DESIGN.ARGS.MORE OPTIONS}={-verbose -debug_log}\n" ;
    print $OFH "prop=run.impl_1.STEPS.ROUTE_DESIGN.TCL.POST=$ABS_DIR/constraints/route_design.post.tcl\n" ;
    print $OFH "prop=run.impl_1.{STEPS.ROUTE_DESIGN.ARGS.MORE OPTIONS}={-verbose}\n" ;
    print $OFH "prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.IS_ENABLED=true\n" ;
    print $OFH "prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=AggressiveExplore\n" ;
    print $OFH "prop=run.impl_1.{STEPS.PHYS_OPT_DESIGN.ARGS.MORE OPTIONS}={-verbose}\n" ;
    print $OFH "prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.TCL.POST=$ABS_DIR/constraints/phys_place_design.post.tcl\n" ;
    print $OFH "prop=run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=NoTimingRelaxation\n" ;
    print $OFH "prop=run.impl_1.{STEPS.ROUTE_DESIGN.ARGS.MORE OPTIONS}={-tns_cleanup}\n" ;
    print $OFH "prop=run.impl_1.STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED=true\n" ;
    print $OFH "prop=run.impl_1.{STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.MORE OPTIONS}={-verbose}\n" ;
    print $OFH "prop=run.impl_1.STEPS.POST_ROUTE_PHYS_OPT_DESIGN.TCL.POST=$ABS_DIR/constraints/phys_route_design.post.tcl\n" ;
    if ( $USE_2LV eq "true" ) {
        print $OFH "prop=run.impl_1.part=xcu50-fsvh2104-2LV-e\n";
    }
}

#sub krnl_x_hbm_conn {
#    my ( $krnl, $num ) = @_ ;
#    my $sp ;
#    given ( $num ) {
#        when ( 3 ) {
#            $sp = $sp . "sp=$krnl.DPU_AXI_0:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_1:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_2:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_I:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_W0:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_W1:HBM[00:31]\n" ;
#        }
#        when ( 4 ) {
#            $sp = $sp . "sp=$krnl.DPU_AXI_0:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_1:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_2:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_3:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_I:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_W0:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_W1:HBM[00:31]\n" ;
#        }
#        when ( 5 ) {
#            $sp = $sp . "sp=$krnl.DPU_AXI_0:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_1:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_2:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_3:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_4:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_I:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_W0:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_W1:HBM[00:31]\n" ;
#        }
#        when ( 6 ) {
#            $sp = $sp . "sp=$krnl.DPU_AXI_0:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_1:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_2:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_3:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_4:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_5:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_I:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_W0:HBM[00:31]\n" ;
#            $sp = $sp . "sp=$krnl.DPU_AXI_W1:HBM[00:31]\n" ;
#        }
#    }
#    return $sp ;
#}

sub connectivity {
    my $vpp_sp ;
    print $OFH "\n[connectivity]\n" ;
    if ( $SLR0 ne 0 ) { 
        print $OFH "nk=${KRNL}_${SLR0}ENGINE:1:dpu_0\n" ;
        #print $OFH krnl_x_hbm_conn ( "${KRNL}_0", $SLR0 ) 
    }
    if ( $SLR1 ne 0 ) { 
        print $OFH "nk=${KRNL}_${SLR1}ENGINE:1:dpu_1\n" ;
        #print $OFH krnl_x_hbm_conn ( "${KRNL}_1", $SLR1 ) 
    }
    if ( $SLR2 ne 0 ) { 
        print $OFH "nk=${KRNL}_${SLR2}ENGINE:1:dpu_2\n" ;
        #print $OFH krnl_x_hbm_conn ( "${KRNL}_2", $SLR2 ) 
    }

    open ( F, "$HBM_PORT_ALLOCATION" ) or die "Failed to Open File $HBM_PORT_ALLOCATION, $!\n" ;
    my $line;
    my $dpu0_engine_num = 0;
    my $dpu1_engine_num = 0;
    my $dpu2_engine_num = 0;
    my @dpu0_wports;
    my $dpu0_wrange_lsb = "";
    my $dpu0_wrange_rsb = "";
    my @dpu1_wports;
    my $dpu1_wrange_lsb = "";
    my $dpu1_wrange_rsb = "";
    my @dpu2_wports;
    my $dpu2_wrange_lsb = "";
    my $dpu2_wrange_rsb = "";
    while ($line=<F>) {
	if ($line =~ /(dpu_)(\d)\/(\w+)\s+(\d+)/) {
	    my $dpu = $1;
	    my $dpu_index = $2;
	    my $dpu_port = $3;
	    my $hbm_index = $4;
	    if ($dpu_index eq "0") {$dpu0_engine_num++;}
	    elsif ($dpu_index eq "1") {$dpu1_engine_num++;}
	    elsif ($dpu_index eq "2") {$dpu2_engine_num++;}
	    else { print "ERROR: dpu_num has exceed 2 - $dpu_index" }

	    #print "Debug: $dpu $dpu_index $dpu_port $hbm_index\n";
	
	    if ($dpu_index eq "0") {
		if ($dpu_port =~ /W\d+/) {;#For weights bank, they should be neighbouring
		    push (@dpu0_wports,"$dpu$dpu_index.$dpu_port");
		    if ($dpu0_wrange_lsb ne "") {
		        $dpu0_wrange_rsb = $hbm_index;
		    } else {
		        $dpu0_wrange_lsb = $hbm_index;
		    }
		} else {
		    my $index = sprintf "%02s",$hbm_index; if ($FORCE_ALL_HBM_ASSIGN eq "true") { $index = "00:31"}
	    	    print $OFH "sp=$dpu$dpu_index.$dpu_port:HBM[$index]\n";
		}
	    } elsif ($dpu_index eq "1") {
		if ($dpu_port =~ /W\d+/) {;#For weights bank, they should be neighbouring
		    push (@dpu1_wports,"$dpu$dpu_index.$dpu_port");
		    if ($dpu1_wrange_lsb ne "") {
		        $dpu1_wrange_rsb = $hbm_index;
		    } else {
		        $dpu1_wrange_lsb = $hbm_index;
		    }
		} else {
		    my $index = sprintf "%02s",$hbm_index; if ($FORCE_ALL_HBM_ASSIGN eq "true") { $index = "00:31"}
	    	    print $OFH "sp=$dpu$dpu_index.$dpu_port:HBM[$index]\n";
		}
	    } elsif ($dpu_index eq "2") {
		if ($dpu_port =~ /W\d+/) {;#For weights bank, they should be neighbouring
		    push (@dpu2_wports,"$dpu$dpu_index.$dpu_port");
		    if ($dpu2_wrange_lsb ne "") {
		        $dpu2_wrange_rsb = $hbm_index;
		    } else {
		        $dpu2_wrange_lsb = $hbm_index;
		    }
		} else {
		    my $index = sprintf "%02s",$hbm_index; if ($FORCE_ALL_HBM_ASSIGN eq "true") { $index = "00:31"}
	    	    print $OFH "sp=$dpu$dpu_index.$dpu_port:HBM[$index]\n";
		}
	    } 

	}
    }
    close (F);
    # deal with weights bank
    foreach (@dpu0_wports) {
	$dpu0_wrange_lsb = sprintf "%02s",$dpu0_wrange_lsb;if ($FORCE_ALL_HBM_ASSIGN eq "true") { $dpu0_wrange_lsb = "00"}
	$dpu0_wrange_rsb = sprintf "%02s",$dpu0_wrange_rsb;if ($FORCE_ALL_HBM_ASSIGN eq "true") { $dpu0_wrange_rsb = "31"}
	print $OFH "sp=$_:HBM[$dpu0_wrange_lsb:$dpu0_wrange_rsb]\n";
    }
    foreach (@dpu1_wports) {
	$dpu1_wrange_lsb = sprintf "%02s",$dpu1_wrange_lsb;if ($FORCE_ALL_HBM_ASSIGN eq "true") { $dpu1_wrange_lsb = "00"}
	$dpu1_wrange_rsb = sprintf "%02s",$dpu1_wrange_rsb;if ($FORCE_ALL_HBM_ASSIGN eq "true") { $dpu1_wrange_rsb = "31"}
	print $OFH "sp=$_:HBM[$dpu1_wrange_lsb:$dpu1_wrange_rsb]\n";
    }
    foreach (@dpu2_wports) {
	$dpu2_wrange_lsb = sprintf "%02s",$dpu2_wrange_lsb;if ($FORCE_ALL_HBM_ASSIGN eq "true") { $dpu2_wrange_lsb = "00"}
	$dpu2_wrange_rsb = sprintf "%02s",$dpu2_wrange_rsb;if ($FORCE_ALL_HBM_ASSIGN eq "true") { $dpu2_wrange_rsb = "31"}
	print $OFH "sp=$_:HBM[$dpu2_wrange_lsb:$dpu2_wrange_rsb]\n";
    }
}

# ##############################################################
# No section names
#
#print $OFH "\nremote_ip_cache=$REMOTE_IP_CACHE\n" ; 
print $OFH "\nkernel_frequency=0:$FREQ0|1:$FREQ1\n" ;
print $OFH "\nuser_ip_repo_paths=../$XO_PATH/DPUCAHX8H_SRC/DPU\n" ;
if  ( $ALVEO eq "u55c" ) {
#print $OFH "\nuser_ip_repo_paths=$DPU_V3E/custom_ip/dpu_clock_gen_2019.1_u280/outputs\n" ;
}
else {
#print $OFH "\nuser_ip_repo_paths=$DPU_V3E/custom_ip/dpu_clock_gen_2019.1_$ALVEO/outputs\n" ;
}
# ##############################################################
# Section starts here
advanced() ;

if ( $STRATEGY eq "Default" ) {
    print $OFH "\n[vivado]\n" ;
    vivado_default() ;
}
else {
    print $OFH "\n[vivado]\n" ;
    vivado_explore() ;
}

connectivity() ;


close $OFH ;
