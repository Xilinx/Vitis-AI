# (C) Copyright 2020 - 2021 Xilinx, Inc.
# SPDX-License-Identifier: Apache-2.0

# Help function
proc help_proc { } {
  puts "Usage: xsct -sdx pfm.tcl -xsa <file>"
  puts "-xsa  <file>       xsa file location"
  puts "-proc <processor>  processor (default: psu_cortexa53)"
  puts "-help              this text"
}

# Increase timeout to avoid sporadic errors
configparams vitis-launch-timeout 400

# Set defaults
set platform "default"
set proc "psv_cortexa72"

# Parse arguments
for { set i 0 } { $i < $argc } { incr i } {
  # xsa file
  if { [lindex $argv $i] == "-xsa" } {
    incr i
    set xsafile [lindex $argv $i]
    set ws [file rootname [file tail $xsafile]]
    set ws "xsct/$ws"
  # processor
  } elseif { [lindex $argv $i] == "-proc" } {
    incr i
    set proc [lindex $argv $i]
  # help
  } elseif { [lindex $argv $i] == "-help" } {
    help_proc
    exit
  # invalid argument
  } else {
    puts "[lindex $argv $i] is an invalid argument"
    exit
  }
}

# helper variables
set platform [file rootname [file tail $xsafile]]
set imagedir "image"
file mkdir $imagedir
set bootdir "boot"
file mkdir $bootdir
set biffile "linux.bif"
set f [open $biffile a]
close $f

# Set workspace
setws $ws

# Create platform
platform create \
	-name $platform \
	-hw $xsafile

# Create domain
domain create \
	-name xrt \
	-os linux \
	-proc $proc

# Configure domain
domain config -image $imagedir
domain config -boot $bootdir
domain config -bif $biffile

# Configure platform
platform config -remove-boot-bsp

# Generate platform
platform -generate
