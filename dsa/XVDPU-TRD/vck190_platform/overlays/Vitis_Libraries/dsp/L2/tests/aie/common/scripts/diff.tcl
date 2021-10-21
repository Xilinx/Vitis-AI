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
# Compare files containing numbers, line by line, value by value.


# Get args
set fileName1             [lindex $argv 0]
set fileName2             [lindex $argv 1]
set fileNameOut           [lindex $argv 2]
if { $::argc >= 4} {
    set tolerance         [lindex $argv 3]
} else {
    # Default tolerenace
    set tolerance         0.001
}
if { $::argc >= 5} {
    # any entry sets absolute tolerance mode on
    set toleranceMode               [lindex $argv 4]
    puts  "toleranceMode = $toleranceMode"
        if {($toleranceMode == "abs")|| ($toleranceMode == "ABS") } {
            set toleranceMode  abs
            # Absolute tolerance Mode
           puts "Using Absolute Tolerance Mode"
        } else {
            # Percentage tolerance Mode
           puts "Using Percentage Tolerance Mode"
        }
} else {
    # Default abs off ( i.e. percentage tolerance mode on )
    set toleranceMode         percent
    puts  "toleranceMode = $toleranceMode"
    # Percentage tolerance Mode
    puts "Using Percentage Tolerance Mode"
}



proc isSame {a b tolerance toleranceMode} {
    if {(($a == 0.0) && ($toleranceMode != "abs"))} {
    # Avoid division by 0
        if {$b == 0.0} {
            set res 1
        } else {
            set res 0
        }
    } else {

        if {($toleranceMode == "abs")} {
            # Absolute tolerance
            set diffVal [expr {abs(($a-$b))}]
        } else {
            # Percentage tolerance
        set diffVal [expr {abs(((double(($a-$b))) / double($a)))}]
        }
        if {$diffVal <= $tolerance} {
            set res 1
        } else {
            set res 0
        }
    }
    return $res
}

set outFile [open $fileNameOut w]
# Write header to file
puts $outFile "Comparing files: "
puts $outFile "File: $fileName1:"
puts $outFile "File: $fileName2:"
puts $outFile "Tolerance ($toleranceMode): $tolerance: "
close $outFile

set fileMismatch 0
set fileLengthMismatch 0
set fileDiffsFound 0
set fileMatchesFound 0
set lineNo 0

# Open files if exist
set fexist1 [file exist $fileName1]
set fexist2 [file exist $fileName2]
if {$fexist1} {
    set inFile1 [open $fileName1 r]
}
if {$fexist2} {
    set inFile2 [open $fileName2 r]
}

# Compare line by line, until EoF.
while {$fexist1 && $fexist2 && [gets $inFile1 line1] != -1} {
    incr lineNo
    if {[gets $inFile2 line2] != -1} {
        set valList1 [split $line1 " "]
        set valList2 [split $line2 " "]
        set indexList2 0
        set printLines 0
        foreach val1  $valList1 {
            # Assume that each files have same number of arguments
            # skip empty spaces at the end of the line
            if {($val1!="")} {
                if {[isSame $val1  [lindex $valList2 $indexList2] $tolerance $toleranceMode] } {
                    # Good, move on
                    incr fileMatchesFound
                } else {
                    # Bad, set out flag to print out diff lines to diff file.
                    set printLines 1
                    # and set the comparison result
                    set fileMismatch 1
                    incr fileDiffsFound
                }
            }
            incr indexList2
        }
        if {$printLines == 1} {
            set outFile [open $fileNameOut a]
            # Write to file
            puts $outFile "Line no: $lineNo:     $line1"
            puts $outFile "Line no: $lineNo:     $line2"
            close $outFile
        }

    } else {
        # Eof reached on file2,
        set fileLengthMismatch 1
        break
    }
}


if {!$fexist1} {
    # File not found
    puts "Error: File not found: $fileName1 "
    set outFile [open $fileNameOut a]
    # Write to file
    puts $outFile "Error: File not found: $fileName1 "
    close $outFile
} elseif {!$fexist2} {
    # File not found
    puts "Error: File not found: $fileName2 "
    set outFile [open $fileNameOut a]
    # Write to file
    puts $outFile "Error: File not found: $fileName2 "
    close $outFile
} elseif {$lineNo == 0} {
    # Empty file
    puts "Error: File empty: $fileName1"
    set outFile [open $fileNameOut a]
    # Write to file
    puts $outFile "Error: File empty: $fileName1"
    close $outFile
} elseif {$fileLengthMismatch != 0} {
    # Empty file
    puts "Error: File length mismatch. Insufficient data in: $fileName2"
    set outFile [open $fileNameOut a]
    # Write to file
    puts $outFile "Error: File length mismatch. Insufficient data in: $fileName2"
    close $outFile
} elseif {$fileMismatch != 0} {
    # Diffs are showing mismatches
    puts "WARNING: Compared files differ. Differences found: $fileDiffsFound"
} else {
    # Good. Put a phrase to a file, which mimics the diff command behaviour.
    set outFile [open $fileNameOut a]
    puts $outFile "Files are identical."
    close $outFile
    puts "INFO: Compared files match."
}



