#
# Copyright 2021 Xilinx, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
set usage "
For generating random stimulus for data files.
tclsh gen_input.tcl <filename> <numSamples> <iterations> \[<seed>\] \[<stim_type>\]
Supported stim_type
    0 = RANDOM
    3 = IMPULSE
    4 = ALL_ONES
    5 = INCR_ONES
    8 = 45 degree spin
    9 = ALT_ZEROES_ONES
";
if { [lsearch $argv "-h"] != -1 } {
    puts $usage
    exit 0
}
set seed 1
set stim_type 0
set dyn_pt_size 0
set max_pt_size_pwr 10
if { $::argc > 2} {
    set filename [lindex $argv 0]
    set fileDirpath [file dirname $filename]
    set window_vsize  [lindex $argv 1]
    set iterations  [lindex $argv 2]
    if {[llength $argv] > 3 } {
        set seed [lindex $argv 3]
    }
    if {[llength $argv] > 4 } {
        set stim_type [lindex $argv 4]
    }
    if {[llength $argv] > 5 } {
        set dyn_pt_size [lindex $argv 5]
    }
    if {[llength $argv] > 6 } {
        set max_pt_size_pwr [lindex $argv 6]
    }
    if {[llength $argv] > 7 } {
        set tt_data [lindex $argv 7]
    }

} else {
    # defaults
    set fileDirpath "./data"
    set filename "$fileDirpath/input.txt"
    set window_vsize 1024
    set iterations 8
    set seed 1
    set stim_type 0 ;# random
    set dyn_pt_size 0
    set max_pt_size_pwr 0
}

set nextRand $seed

proc srand {seed} {
    set nextRand $seed
}

proc randInt16 {seed} {
    set nextRand [expr {($seed * 1103515245 + 12345)}]
    return [expr (($nextRand % 65536) - 32768)]
}
proc randInt32 {seed} {
    set nextRand [expr {($seed * 1103515245 + 12345)}]
    return [expr (int($nextRand))]
}
# If directory already exists, nothing happens
file mkdir $fileDirpath
set output_file [open $filename w]
set headRand [srand $seed]
set blank_entry "0 0"
#ensure that a sim stall doesn't occur because of insufficient data (yes that would be a bug)
set overkill 1
set padding 0
set pt_size_pwr $max_pt_size_pwr+1
set iters 1
set samples $window_vsize
for {set iter_nr 0} {$iter_nr < [expr ($iterations*$overkill)]} {incr iter_nr} {
    if {$dyn_pt_size == 1} {
        set headRand [randInt16 $headRand]
        #use fields of the random number to choose FFT_NIFFT and PT_SIZE_PWR. Choose a legal size
        set fft_nifft [expr (($headRand >> 14) % 2)]
#        set pt_size_pwr [expr (($headRand % ($max_pt_size_pwr-4))+4)]
        set pt_size_pwr [expr ($pt_size_pwr - 1)]
        if {$pt_size_pwr < 5} {
            set pt_size_pwr $max_pt_size_pwr
        }
        set header_size 4
        if {$tt_data eq "cint16"} {
            set header_size 8
        }
        puts $output_file "$fft_nifft 0"
        puts $output_file "$pt_size_pwr 0"
        # pad. This loops starts at 2 because 2 samples have already been written
        for {set i 2} {$i < $header_size} {incr i} {
            puts $output_file $blank_entry
        }
        set samples [expr (1 << $pt_size_pwr)]
        set padding 0
        if { $pt_size_pwr < $max_pt_size_pwr } {
            set padding [expr ((1 << $max_pt_size_pwr) - $samples)]
        }
        set iters [expr (($window_vsize-$header_size)/($samples+$padding))]
    }
    for {set winSplice 0} {$winSplice < $iters} {incr winSplice} {
        for {set sample_idx 0} {$sample_idx < [expr ($samples)]} {incr sample_idx} {
            set sample_nr [expr (($samples * $iter_nr)+$sample_idx)]
            if { $stim_type == 4 || ($stim_type == 3 && $sample_nr == 0) || ($stim_type == 9 && $sample_nr%2 == 1) } {
                set nextRand 1
            } elseif { ($stim_type == 3 && $sample_nr != 0) || $stim_type == 9 } {

                set nextRand 0
            } elseif { ($stim_type == 5) } {
                set nextRand [expr $nextRand+1]
            } elseif { ($stim_type == 8) } {
                if {$sample_idx % 8 == 0 } {
                    set nextRand 8192
                }  elseif {$sample_idx % 8 == 1 || $sample_idx % 8 == 7 } {
                    set nextRand 5793
                }  elseif {$sample_idx % 8 == 2 || $sample_idx % 8 == 6 } {
                    set nextRand 0
                }  elseif {$sample_idx % 8 == 3 || $sample_idx % 8 == 5 } {
                    set nextRand -5793
                }  else {
                    set nextRand -8192
                }
            } else {
                set nextRand [randInt16 $nextRand]
            }
            puts -nonewline $output_file "$nextRand "
            if { $stim_type == 4 || ($stim_type == 9 && $sample_nr%2 == 1)} {
                set nextRand 1
            } elseif { ($stim_type == 3) || ($stim_type == 9)} {
                set nextRand 0
            } elseif { ($stim_type == 5) } {
                set nextRand $nextRand
            } elseif { ($stim_type == 8) } {
                if {$sample_idx % 8 == 0 | $sample_idx % 8 == 4 } {
                    set nextRand 0
                }  elseif {$sample_idx % 8 == 1 || $sample_idx % 8 == 3 } {
                    set nextRand 5793
                }  elseif {$sample_idx % 8 == 2 } {
                    set nextRand 8192
                }  elseif {$sample_idx % 8 == 5 || $sample_idx % 8 == 7 } {
                    set nextRand -5793
                }  else {
                    set nextRand -8192
                }
            } else {
                set nextRand [randInt16 $nextRand]
            }
            puts $output_file $nextRand
        }
        #padding is only non-zero for dynamic point size, so no need to clause with dyn_pt_size
        for {set sample_idx 0} {$sample_idx < [expr ($padding)]} {incr sample_idx} {
            set padsample -1
            puts -nonewline $output_file "$padsample "
            puts $output_file $padsample
        }
    }
}

