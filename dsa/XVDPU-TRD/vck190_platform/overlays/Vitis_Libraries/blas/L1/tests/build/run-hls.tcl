####################
# A example to build and debug vivado_hls project
# vivado_hls -f ./build/run-hls.tcl "runCsim 1 runRTLsynth 0 runRTLsim 0 part vu9p dataType double dataWdith 64 resDataType int size 8192 3 logParEntries 4 opName amax runArgs '../out_test/data/app.bin'"
# navigate to csim/build and run
# gdb --args ./csime.exe path_to_app_bin/app.bin 8192
####################
set pwd [pwd]
set pid [pid]

set VIVADO_PATH $::env(XILINX_VIVADO)

set GCC_VERSION 6.2.0
set GCC_PATH "$VIVADO_PATH/tps/lnx64"
set BOOST_INCLUDE "$VIVADO_PATH/tps/boost_1_64_0"
set BOOST_LIB "$VIVADO_PATH/lib/lnx64.o"

set TESTDIR [lindex $argv 2]
set PARAM_FILE [lindex $argv 3]
set DIRECTIVE_FILE [lindex $argv 4]
set RUNARGS [lindex $argv 5]
source $TESTDIR/$PARAM_FILE

puts "Final CONFIG"
set OPT_FLAGS "-std=c++11 "
foreach o [lsort [array names opt]] {
  if { [string match "run*" $o] == 0 } {
    puts "  Using CONFIG  $o  [set opt($o)]"
    append OPT_FLAGS [format {-D BLAS_%s=%s } $o $opt($o)]
  }
}

set CFLAGS_K "-I$TESTDIR -I$TESTDIR/../include/hw -I$TESTDIR/hw -I$TESTDIR/../include/hw/xf_blas  -g -O0 $OPT_FLAGS"
set CFLAGS_H "$CFLAGS_K -I$TESTDIR -I$TESTDIR/../include/hw -I$TESTDIR/../include/hw/xf_blas -I$TESTDIR/hw -I$TESTDIR/sw/include -I$TESTDIR/../.. -I$TESTDIR/hw -I$BOOST_INCLUDE"

set proj_dir [format prj_hls_%s  $opt(part) ]
open_project $proj_dir -reset
set_top uut_top 
add_files $TESTDIR/$opt(path)/uut_top.cpp -cflags "$CFLAGS_K"
add_files -tb $TESTDIR/sw/src/test.cpp -cflags "$CFLAGS_H"
open_solution sol -reset

source $TESTDIR/$DIRECTIVE_FILE

set_part $opt(part)

create_clock -period 3.333333 -name default


if {$opt(runCsim)} {
  puts "***** C SIMULATION *****"
  csim_design -ldflags "-L$BOOST_LIB -lboost_iostreams -lz -lrt -L$GCC_PATH/$GCC_VERSION/lib64 -lstdc++ -Wl,--rpath=$BOOST_LIB" -argv "$TESTDIR/$RUNARGS"
}

if {$opt(runRTLsynth)} {
  puts "***** C/RTL SYNTHESIS *****"
  csynth_design
  if {$opt(runRTLsim)} {
    puts "***** C/RTL SIMULATION *****"
    cosim_design -trace_level all -ldflags "-L$BOOST_LIB -lboost_program_options -lrt" -argv "$TESTDIR/$RUNARGS"
  }
}

exit
