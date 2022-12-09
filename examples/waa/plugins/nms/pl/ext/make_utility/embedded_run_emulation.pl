#
## Copyright 2020 Xilinx, Inc.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##

use strict;
use warnings;
use Expect;
my $k=0;
my $j=0;
my $i=0;
my $EMULATOR_COMMAND=$ARGV[0];
my $EXECUTABLE=$ARGV[1];
#my $XCLBIN=$ARGV[2];
my $command;
my $status;
my $timeoutlevel;
my $delay;
my @timeout = (3,10,20,30,45,60,90,120,180,300,420,600,720,1080,1440,2160,2880,4320,7200,11520);
my $exit_val=0;

#$command="launch_emulator -no-reboot -runtime ocl -t hw_emu -sd-card-image sd_card.img -device-family Ultrascale";
$command=$EMULATOR_COMMAND;

my($exp)  = new Expect; #Set exp for Expect
my($exp1) = new Expect; #Set exp for Expect



if ($ARGV[2] eq '') { 
     $status="PASS";
}
else {
     $status=$ARGV[2];
}


if ($ARGV[3] eq '') { 
     $delay=100;
}
else {
     $timeoutlevel=$ARGV[3];    
     $delay= $timeout[int($timeoutlevel)-1];
}


$exp-> spawn("$command")
or die "ERROR: Cannot spawn $command: $!\n";

#| Detect if linux booted on Zynq:
my $booted = 0;
 while ($i<15) {
    if( $exp->expect(180, '-re', 'root@.*:.*#')) {
        $booted=1;
        printf "\nINFO: Linux kernel booted successfully\n";
        last;
    }
    
    if( $exp->expect(120, 'ZynqMP>')) {
        printf "\nERROR: Linux kernel boot failed\n";
        $exit_val=1;
        print "\nINFO: Exiting QEMU\n";
        #sleep(10);
        $exp-> send("\x01");
        $exp-> send("x");
        #$exp-> send(undef);
        #$exp->soft_close();
        sleep(20);
        exit($exit_val);
    }
    
    if( $exp->expect(30, 'Simulation engine failed to start')) {
        printf "\nERROR: [Simtcl 6-50] Simulation engine failed to start: Simulation exited with status non-0 code\n";
        $exit_val=1;
        print "\nINFO: Exiting QEMU\n";
        sleep(10);
        #$exp-> send("\x01");
        #$exp-> send("x");
        #$exp-> send(undef);
        $exp->soft_close();
        exit($exit_val);
    }
    
	sleep(10);
	
#	$exp-> send( "echo 'INFO: Embedded host run completed.'");
#   sleep(10);
	#last;
    
    $i=$i+1;

  }

if( $booted == 0) {
    printf "\nERROR: It appears that booting hanged. Exiting QEMU.\n";
    #sleep(10);
    $exp-> send("\x01");
    $exp-> send("x");
    sleep(20);
    #$exp-> send(undef);
    #$exp->soft_close();
    $exit_val = 2;
    exit($exit_val);
}
  
#| Execute commands if system is up and running:
$exp-> send( "mount /dev/mmcblk0p1 /mnt\r");
$exp-> send( "cd /mnt\r");
$exp-> send( "$EXECUTABLE\r");

#| Check for finish of the host app:
#while ($j<20)
#{
if( $exp-> expect(int($delay)*45, "Embedded host run completed" )) {
    $k=1;
    #last;
}
    
	#$j=$j+1;
    #sleep(10);
#}



    #if( $xp-> expect(15, $status)) {
if($k==1) {
    #print "\nINFO: Embedded host run completed.\n";
    #$exp-> send( "echo 'INFO: Embedded host run completed.'");
    $exit_val=0;
    #last;
} else {
    print "\nERROR: Host application did not complete - pass/end string not found. Exiting as a failure\n";
    $exp-> send("\x01");
    sleep(2);
    $exp-> send("x");
    sleep(5);
    $exp->soft_close();
    sleep(5);
    $exit_val=1;
    exit($exit_val);
  }
#if($j==3)
#{
#print "TEST PASSED String Not Found, Hence exiting ";
      #  exit(1);
#                $exit_val=1;
      #
#}
#if( $exp-> expect(100, "TEST FAILED")) {
#    print "Embedded Host Run Failed";
#	exit(1);
#  }

# if( $i < 3 ){
    # printf "INFO: Linux kernel booted successfully\n";
# }else{
    # printf "ERROR: Linux kernel boot failed\n";
    # printf "\n";
        # $exit_val=1;   
# }

print "\nINFO: Exiting QEMU \n";
sleep(1);
print "\nINFO: Running reboot \n";
my $start_time = time();
$exp->send("reboot -f\r");

if( $exp-> expect(600, '-re', "Emulation ran successfully" )) {
    my $end_time = time();
    my $diff_time = $end_time - $start_time;
    print "INFO: Found reboot finishing string. Reboot took: ${diff_time} seconds.\n";
    sleep(20);
    $exp->soft_close();
} else {
    print "\nWARNING: Reboot did not finish in 10mins.\n";
}


#sleep(120);
#$exp->soft_close();
#while( not defined($exp->exitstatus()) ) {
    #| Wait 2 seconds to check if exit status is defined:
#    sleep(2);
#}
#$exp-> send("\x01");
#sleep(2);
#$exp-> send("x");
#sleep(5);
#$exp-> send(undef);
#$exp->soft_close();
#sleep(5);
exit($exit_val);
