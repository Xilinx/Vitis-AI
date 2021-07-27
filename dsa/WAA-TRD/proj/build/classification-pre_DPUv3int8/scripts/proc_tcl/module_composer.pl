#! /usr/bin/perl
#*****************************************************************************
#
# DESCRIPTION:
#
# FEATURES:
#
# 
#
# AUTHORS:
#    Shawn Fang (shaoxia.fang@deephi.tech)
#    v1.0 2017-04-07
#   
#-----------------------------------------------------------------------------
#                            REVISION HISTORY
#   
# Usage: module_composer.pl -i [module.xml] -output_dir(optional) [OUTPUT_DIR]
#****************************************************************************/
#
#
use warnings;
#use strict;
#use FindBin;
use XML::Simple;
use Data::Dumper;


$main::log_str="";


&get_paras;
&read_xml;

&link;



sub read_xml{
        my $xml=XMLin($main::infile,ForceArray=>1);
        #my $xml=XMLin("/home/shaoxia/work/dpu_v2_fpga/rtl/hwpart/xczu9eg-ffvb1156-2-i-es2/board.xml",ForceArray=>1);
        #$main::log_str.="Read xml:".$main::infile."--------------------\n\n";
        $main::log_str.="Elaborating...\n\n";
        $main::log_str.=Dumper($xml)."\n\n";
        $main::log_str.="Elaborating Done-----------------------\n\n";
        
        $main::top_design{name}=&trim($xml->{module});
        $main::log_str.="TOP name=".$main::top_design{name}."\n\n";
        $main::log_str.="Extracting TOP ports-----------\n\n";
        
        my $i;my $j;my $k;
        $i=0;$j=0;$k=0;
        
        if($xml->{ports}->[0]->{content}){
                $main::top_design{ports}=&extract_ports($xml->{ports}->[0]->{content},"str");
                while($xml->{ports}->[0]->{pattern}->[$i]){
                        $main::top_design{ports}{pattern_prefix}[$i]=&trim($xml->{ports}->[0]->{pattern}->[$i]->{prefix});
                        $main::top_design{ports}{pattern_suffix}[$i]=&trim($xml->{ports}->[0]->{pattern}->[$i]->{suffix});
                        $main::top_design{ports}{pattern_prefix}[$i]="" if(!$main::top_design{ports}{pattern_prefix}[$i]);
                        $main::top_design{ports}{pattern_suffix}[$i]="" if(!$main::top_design{ports}{pattern_suffix}[$i]);
                        $main::log_str.= sprintf("Port Pattern[%d]: prefix=%s ; suffix=%s \n\n",$i,$main::top_design{ports}{pattern_prefix}[$i],$main::top_design{ports}{pattern_suffix}[$i]);
                        $i++;
                }

                $i=0;
                while($xml->{ports}->[0]->{regex}->[$i]){
                        $main::top_design{ports}{regex}{pattern_src}[$i]=&trim($xml->{ports}->[0]->{regex}->[$i]->{pattern_src});
                        $main::top_design{ports}{regex}{pattern_dst}[$i]=&trim($xml->{ports}->[0]->{regex}->[$i]->{pattern_dst});
                        $main::top_design{ports}{regex}{pattern_src}[$i]="" if(!$main::top_design{ports}{regex}{pattern_src}[$i]);
                        $main::top_design{ports}{regex}{pattern_dst}[$i]="" if(!$main::top_design{ports}{regex}{pattern_dst}[$i]);
                        $main::log_str.= sprintf("Port regex[%d]: pattern_src=%s ; pattern_dst=%s \n\n",$i,$main::top_design{ports}{regex}{pattern_src}[$i],$main::top_design{ports}{regex}{pattern_dst}[$i]);
                        $i++;
                }
        }
        else{
                $main::top_design{ports}=&extract_ports($xml->{ports}->[0],"str");
        }
        
        $i=0;$j=0;$k=0;
        while($xml->{background}->[0]->{submodule}->[$i]){
                
                $main::top_design{background}{submodule}{name}[$i]=&trim($xml->{background}->[0]->{submodule}->[$i]->{module});
                $main::top_design{background}{submodule}{port_src}[$i]=$xml->{background}->[0]->{submodule}->[$i]->{port_src};
                $main::top_design{background}{submodule}{port_src}[$i]=~s/\$\{([^{}]+)\}/$ENV{$1}/g;
		#print $main::top_design{background}{submodule}{port_src}[$i]."\n";
		$main::top_design{background}{submodule}{ports}[$i]=&extract_ports($main::top_design{background}{submodule}{port_src}[$i],"file");
                $main::log_str.= sprintf("Background: submodule[%d]: name=%s , port_src_file=%s\n\n",$i,$main::top_design{background}{submodule}{name}[$i],$main::top_design{background}{submodule}{port_src}[$i]);
                $j=0;
                while($xml->{background}->[0]->{submodule}->[$i]->{pattern}->[$j]){

                        $main::top_design{background}{submodule}{pattern_prefix}[$i][$j]=&trim($xml->{background}->[0]->{submodule}->[$i]->{pattern}->[$j]->{prefix});
                        $main::top_design{background}{submodule}{pattern_suffix}[$i][$j]=&trim($xml->{background}->[0]->{submodule}->[$i]->{pattern}->[$j]->{suffix});
                        
                        $main::top_design{background}{submodule}{pattern_prefix}[$i][$j]="" if(!$main::top_design{background}{submodule}{pattern_prefix}[$i][$j]);
                        $main::top_design{background}{submodule}{pattern_suffix}[$i][$j]="" if(!$main::top_design{background}{submodule}{pattern_suffix}[$i][$j]);
                        $main::log_str.= sprintf("Background: submodule[%d]->pattern[%d]: prefix=%s,suffix=%s\n\n",$i,$j,$main::top_design{background}{submodule}{pattern_prefix}[$i][$j],$main::top_design{background}{submodule}{pattern_suffix}[$i][$j]);
                        $j++;
                }

                $i++;
        }
        
        $i=0;$j=0;$k=0;
        while($xml->{include}->[$i]){
                if($xml->{include}->[$i]->{file}->[0]->{f_name}){
                        $main::top_design{include}{name}[$j]=$xml->{include}->[$i]->{file}->[0]->{f_name};
                        $main::log_str.= sprintf("Inlcude[%d]: name=%s \n\n",$i,$main::top_design{include}{name}[$j]);
                        $i++;
                        $j++;
                }
                else{
                        $main::log_str.= sprintf("Inlcude File Empty: seq=%d\n\n",$i);
                        $i++;
                }
        }
        
        $i=0;$j=0;$k=0;
        while($xml->{instance}->[$i]){
                 
                 
                $main::top_design{instance}{module}[$i]=&trim($xml->{instance}->[$i]->{module});
                $main::top_design{instance}{num}[$i]=&trim($xml->{instance}->[$i]->{num});
                
                $main::log_str.= sprintf("Instance[%d]: module=%s , num=%s\n\n",$i, $main::top_design{instance}{module}[$i],$main::top_design{instance}{num}[$i]);
                $j=0;
                while($xml->{instance}->[$i]->{regex}->[$j]){
                        $main::top_design{instance}{regex}{pattern_src}[$i][$j]=&trim($xml->{instance}->[$i]->{regex}->[$j]->{pattern_src});
                        $main::top_design{instance}{regex}{pattern_dst}[$i][$j]=&trim($xml->{instance}->[$i]->{regex}->[$j]->{pattern_dst});
                        $main::top_design{instance}{regex}{pattern_src}[$i][$j]="" if(!$main::top_design{instance}{regex}{pattern_src}[$i][$j]);
                        $main::top_design{instance}{regex}{pattern_dst}[$i][$j]="" if(!$main::top_design{instance}{regex}{pattern_dst}[$i][$j]);
                        $main::log_str.= sprintf("Instance[%d]->regex[%d]: pattern_src=%s , pattern_dst=%s\n\n",$i,$j,$main::top_design{instance}{regex}{pattern_src}[$i][$j],$main::top_design{instance}{regex}{pattern_dst}[$i][$j]);
                        $j++;
                }
                $k=0;
                while($xml->{instance}->[$i]->{para}->[$k]){
                        $main::top_design{instance}{para}{name}[$i][$k]=&trim($xml->{instance}->[$i]->{para}->[$k]->{p_name});
                        $main::top_design{instance}{para}{value}[$i][$k]=&trim($xml->{instance}->[$i]->{para}->[$k]->{value});
                        $main::log_str.= sprintf("Instance[%d]->para[%d]: name=%s , value=%s\n\n",$i,$k,$main::top_design{instance}{para}{name}[$i][$k],$main::top_design{instance}{para}{value}[$i][$k]);
                        $k++;
                }

                $i++;
        }
        
        $i=0;$j=0;$k=0;
        while($xml->{connect}->[$i]){
                $main::top_design{connect}{pid_src}[$i]=&trim($xml->{connect}->[$i]->{pid_src});
                $main::top_design{connect}{pid_dst}[$i]=&trim($xml->{connect}->[$i]->{pid_dst});
                my @pid_src_array=split(/\./,$main::top_design{connect}{pid_src}[$i]);
                my @pid_dst_array=split(/\./,$main::top_design{connect}{pid_dst}[$i]);
                
                $main::top_design{connect}{src_module}[$i]=&trim($pid_src_array[0]);
                $main::top_design{connect}{dst_module}[$i]=&trim($pid_dst_array[0]);
                
                if($main::top_design{connect}{src_module}[$i] eq "ports" || $main::top_design{connect}{src_module}[$i] eq "port")
                {
                        $main::top_design{connect}{src_inst_id}[$i]="N/A";
                        $main::top_design{connect}{src_pattern_id}[$i]=&trim($pid_src_array[1]);
                }
                else{
                        $main::top_design{connect}{src_inst_id}[$i]=&trim($pid_src_array[1]);
                        $main::top_design{connect}{src_pattern_id}[$i]=&trim($pid_src_array[2]);
                }
                
                if($main::top_design{connect}{dst_module}[$i] eq "ports" || $main::top_design{connect}{dst_module}[$i] eq "port")
                {
                        $main::top_design{connect}{dst_inst_id}[$i]="N/A";
                        $main::top_design{connect}{dst_pattern_id}[$i]=&trim($pid_dst_array[1]);
                }
                else{
                        $main::top_design{connect}{dst_inst_id}[$i]=&trim($pid_dst_array[1]);
                        $main::top_design{connect}{dst_pattern_id}[$i]=&trim($pid_dst_array[2]);
                }
                
                
                $main::log_str.= sprintf("Connect[%d]: pid_src=%s , pid_dst=%s\n\n",$i,$main::top_design{connect}{pid_src}[$i],$main::top_design{connect}{pid_dst}[$i]);
                $j=0;
                while($xml->{connect}->[$i]->{regex}->[$j]){
                        $main::top_design{connect}{regex}{pattern_src}[$i][$j]=$xml->{connect}->[$i]->{regex}->[$j]->{pattern_src};
                        $main::top_design{connect}{regex}{pattern_dst}[$i][$j]=$xml->{connect}->[$i]->{regex}->[$j]->{pattern_dst};
                        $main::top_design{connect}{regex}{pattern_src}[$i][$j]="" if(!$main::top_design{connect}{regex}{pattern_src}[$i][$j]);
                        $main::top_design{connect}{regex}{pattern_dst}[$i][$j]="" if(!$main::top_design{connect}{regex}{pattern_dst}[$i][$j]);
                        $main::log_str.= sprintf("Connect[%d]->Regex[%d]: pattern_src=%s , pattern_dst=%s\n\n",$i,$j,$main::top_design{connect}{regex}{pattern_src}[$i][$j],$main::top_design{connect}{regex}{pattern_dst}[$i][$j]);
                        $j++;
                }

                $i++;
        }
        $main::top_design{usercontent}=$xml->{usercontent}->[0];
        $main::log_str.= sprintf("Usercontent: \n%s\n\n",$main::top_design{usercontent});
        
        $main::log_str.="\n\nRead XML DONE.------------------------\n\n";
}

sub component_check{
        
}

sub link{

        &component_check;
        
        my $top_str="";
        #1. TOP Ports-------------
        $top_str.="\n\n//This File is Generated at ".`date`."//by ".`whoami`."\n";
        my $top_ports=&trim($main::top_design{ports});
        $top_ports=~s/;/,/g;
        $top_ports=substr($top_ports,0,length($top_ports)-1);
        if($main::top_design{ports}{regex}{pattern_src}){
               my $regex_num=scalar (@{ $main::top_design{ports}{regex}{pattern_src} });
               for(my $k=0;$k<$regex_num;$k++){
                   my $regex_src=$main::top_design{ports}{regex}{pattern_src}[$k];
                   my $regex_dst=$main::top_design{ports}{regex}{pattern_dst}[$k];
                   $top_ports=~s/$regex_src/$regex_dst/g;     
               }
        }
        if($top_ports eq ""){
                $top_str.=sprintf("module %s;\n",$main::top_design{name});
        }
        else{
                $top_str.=sprintf("module %s (\n%s\n);\n",$main::top_design{name},$top_ports);
        }
        $top_str.="\n";
        foreach(@{$main::top_design{include}{name}}){
                $top_str.="`include \"$_\"\n";
        }
        $top_str.="\n";

        #2. wire declarations------------------
        my $inst_module_num=scalar @{ $main::top_design{instance}{module} };
        for(my $i=0;$i<$inst_module_num;$i++){
                my $inst_dup_num=$main::top_design{instance}{num}[$i];
                my $module_name=$main::top_design{instance}{module}[$i];
                my $sub_ports=&get_ports_of_submodule($module_name);
                
                
                for(my $j=0;$j<$inst_dup_num;$j++){
                       my $wires=&gen_wire_declaration($module_name,
                                 $inst_dup_num==1?-1:$j,
                                 $sub_ports
                                );
                       if($main::top_design{instance}{regex}{pattern_src}[$i]){
                               my $regex_num=scalar (@{ $main::top_design{instance}{regex}{pattern_src}[$i] });
                               for(my $k=0;$k<$regex_num;$k++){
                                   my $regex_src=$main::top_design{instance}{regex}{pattern_src}[$i][$k];
                                   my $regex_dst=$main::top_design{instance}{regex}{pattern_dst}[$i][$k];
                                   $wires=~s/$regex_src/$regex_dst/g;     
                               }
                       }
                       $top_str.=$wires;
                }
        }
        $top_str.="\n";
        #3.wire connections-------------------
        my $connect_num=scalar @{ $main::top_design{connect}{pid_src} };
        for(my $i=0;$i<$connect_num;$i++){
                my $src_is_port=($main::top_design{connect}{src_module}[$i] eq "port" || $main::top_design{connect}{src_module}[$i] eq "ports") ? 1 :0;
                my $dst_is_port=($main::top_design{connect}{dst_module}[$i] eq "port" || $main::top_design{connect}{dst_module}[$i] eq "ports") ? 1 :0;
        
                $top_str.=sprintf("//connect:%s<->%s\n",$main::top_design{connect}{pid_src}[$i],$main::top_design{connect}{pid_dst}[$i]);
                my $src_inst_num=$src_is_port? "N/A" :&get_inst_num_of_submodule($main::top_design{connect}{src_module}[$i]);
                my $dst_inst_num=$dst_is_port? "N/A" :&get_inst_num_of_submodule($main::top_design{connect}{dst_module}[$i]);
                $top_str.=sprintf("//src_inst_num=%s,dst_inst_num=%s\n",$src_inst_num,$dst_inst_num);
                my $src_module_id=$src_is_port? "N/A" :&get_id_of_submodule($main::top_design{connect}{src_module}[$i]);
                my $dst_module_id=$dst_is_port? "N/A" :&get_id_of_submodule($main::top_design{connect}{dst_module}[$i]);
                my $src_pattern_id=$main::top_design{connect}{src_pattern_id}[$i];
                my $dst_pattern_id=$main::top_design{connect}{dst_pattern_id}[$i];
                $top_str.=sprintf("//src_pattern_id=%s,dst_pattern_id=%s\n",$src_pattern_id,$dst_pattern_id);
                my $src_pattern_prefix=$src_is_port?$main::top_design{ports}{pattern_prefix}[$src_pattern_id]: $main::top_design{background}{submodule}{pattern_prefix}[$src_module_id][$src_pattern_id];
                $src_pattern_prefix=""  if(!$src_pattern_prefix);
                my $src_pattern_suffix=$src_is_port?$main::top_design{ports}{pattern_suffix}[$src_pattern_id]: $main::top_design{background}{submodule}{pattern_suffix}[$src_module_id][$src_pattern_id];
                $src_pattern_suffix=""  if(!$src_pattern_suffix);
                my $dst_pattern_prefix=$dst_is_port?$main::top_design{ports}{pattern_prefix}[$dst_pattern_id]:$main::top_design{background}{submodule}{pattern_prefix}[$dst_module_id][$dst_pattern_id];
                $dst_pattern_prefix=""  if(!$dst_pattern_prefix);
                my $dst_pattern_suffix=$dst_is_port?$main::top_design{ports}{pattern_suffix}[$dst_pattern_id]:$main::top_design{background}{submodule}{pattern_suffix}[$dst_module_id][$dst_pattern_id];
                $dst_pattern_suffix=""  if(!$dst_pattern_suffix);
                $top_str.=sprintf("//src_module_id=%s,dst_module_id=%s\n",$src_module_id,$dst_module_id);
                $top_str.=sprintf("//src_inst_id=%s,dst_inst_id=%s\n",$main::top_design{connect}{src_inst_id}[$i],$main::top_design{connect}{dst_inst_id}[$i]);
                $top_str.=sprintf("//src_pattern_prefix=%s,src_pattern_suffix=%s\n", $src_pattern_prefix,$src_pattern_suffix);
                $top_str.=sprintf("//dst_pattern_prefix=%s,dst_pattern_suffix=%s\n",$dst_pattern_prefix,$dst_pattern_suffix);
                my $src_wires_prefix=$src_is_port? ""
                              :($src_inst_num==1?
                                ($main::top_design{connect}{src_module}[$i]."_"):
                                ($main::top_design{connect}{src_module}[$i]."_".$main::top_design{connect}{src_inst_id}[$i]."_"));
                my $dst_wires_prefix=$dst_is_port? ""
                              :($dst_inst_num==1?
                                ($main::top_design{connect}{dst_module}[$i]."_"):
                                ($main::top_design{connect}{dst_module}[$i]."_".$main::top_design{connect}{dst_inst_id}[$i]."_"));
                my $src_ports=$src_is_port? $top_ports:$main::top_design{background}{submodule}{ports}[$src_module_id];
                my $dst_ports=$dst_is_port? $top_ports:$main::top_design{background}{submodule}{ports}[$dst_module_id];
                
                my $connect_str=&connect_module1_module2($src_ports,
                        $src_wires_prefix,
                        $src_pattern_prefix,$src_pattern_suffix,
                        $dst_ports,
                        $dst_wires_prefix,
                        $dst_pattern_prefix,$dst_pattern_suffix,
                        $src_is_port? "reverse_assign":"normal"
                   );
                if($main::top_design{connect}{regex}{pattern_src}[$i]){
                               my $regex_num=scalar (@{ $main::top_design{connect}{regex}{pattern_src}[$i] });
                               for(my $k=0;$k<$regex_num;$k++){
                                   my $regex_src=$main::top_design{connect}{regex}{pattern_src}[$i][$k];
                                   my $regex_dst=$main::top_design{connect}{regex}{pattern_dst}[$i][$k];
                                   $connect_str=~s/$regex_src/$regex_dst/g;     
                               }
                       }
                $top_str.=$connect_str."\n\n";
        }
        
        #4.instantiate-------------
        for(my $i=0;$i<$inst_module_num;$i++){
                my $inst_dup_num=$main::top_design{instance}{num}[$i];
                my $module_name=$main::top_design{instance}{module}[$i];
                my $sub_ports=&get_ports_of_submodule($module_name);
                my $paras="";
                if($main::top_design{instance}{para}{name}[$i]){
                        my $para_num=scalar @{ $main::top_design{instance}{para}{name}[$i] };
                        for(my $j=0;$j<$para_num;$j++){
                                if($j==$para_num-1){
                                        $paras.=sprintf("    .%s(%s)\n",$main::top_design{instance}{para}{name}[$i][$j],$main::top_design{instance}{para}{value}[$i][$j]);
                                }
                                else{
                                        $paras.=sprintf("    .%s(%s),\n",$main::top_design{instance}{para}{name}[$i][$j],$main::top_design{instance}{para}{value}[$i][$j]);
                                }
                        }
                        if($para_num>0){
                                $paras=sprintf("#(\n$paras)");
                        }
                }
                for(my $j=0;$j<$inst_dup_num;$j++){
                        $top_str.=&gen_instance($module_name,
                                ($inst_dup_num>1?$j:-1),
                                $sub_ports, 
                                $paras
                        );
                }
        }
        #5.user content-------------
        $top_str.=$main::top_design{usercontent}."\n";
        
        #6. endmodule
        $top_str.="endmodule\n\n\n";
        
        my $output_file=sprintf("%s/%s.v",$main::output_dir,$main::top_design{name});
        open FILE, "> $output_file" or die "Cannot open file $output_file: $! ";
	print FILE $top_str;
	close FILE;
	print "\nModule Composer Done!\nOutput composed file:\t$output_file \n";
	
	my $log_file=sprintf("%s/%s.log",$main::output_dir,$main::top_design{name});
        open FILE, "> $log_file" or die "Cannot open file $log_file: $! ";
	print FILE $log_str;
	close FILE;
	print "Logfile:\t$log_file \n\n";
}
sub connect_module1_module2{
	my $m1_ports=$_[0];
	my $m1_wires_prefix=$_[1];
	my $m1_filter_pattern_p=$_[2];
	my $m1_filter_pattern_s=$_[3];

	my $m2_ports=$_[4];
	my $m2_wires_prefix=$_[5];
	my $m2_filter_pattern_p=$_[6];
	my $m2_filter_pattern_s=$_[7];
	
	my $reverse_assign=$_[8];
	
	$m1_filter_pattern_p="" if(!$m1_filter_pattern_p);
	$m1_filter_pattern_s="" if(!$m1_filter_pattern_s);
	$m2_filter_pattern_p="" if(!$m2_filter_pattern_p);
	$m2_filter_pattern_s="" if(!$m2_filter_pattern_s);

	my $str="";
	#print "debug1[$m1_filter_pattern_p,$m1_filter_pattern_s]:$m1_ports\n\n" if $m1_filter_pattern_p=~/irq_/;
	foreach my $line(split("\n",$m1_ports)){
		
		if($line=~/[\]\s\t]$m1_filter_pattern_p(\w+)$m1_filter_pattern_s\s*[\t\s\n;,]*\z/){
			$flag_m1=$1;
			#print  "debug3[$m1_filter_pattern_p,$m1_filter_pattern_s]:$line\n\n" if $m1_filter_pattern_p=~/irq_/;
			my $direction=1;
			if($line=~/^\s*input/){
				$direction=0;
			}
			if($reverse_assign){
			        $direction=(1-$direction) if($reverse_assign eq "reverse_assign");
			}
			my $found=0;
			foreach my $line2(split("\n",$m2_ports)){
				#print "debug2:$line;$line2\n" if $line=~/irq_/ && $line2=~/irq_/;
				if($line2=~/[\]\s\t]$m2_filter_pattern_p(\w+)$m2_filter_pattern_s\s*[\t\s\n;,]*\z/){
					$flag_m2=$1;
					if(lc($flag_m1) eq lc($flag_m2)){
					        if($direction==0){
						
						        $str.="assign $m1_wires_prefix"."$m1_filter_pattern_p$flag_m1$m1_filter_pattern_s=$m2_wires_prefix"."$m2_filter_pattern_p$flag_m2$m2_filter_pattern_s;//$line//$flag_m1//$line2\n";
					
					        }
					        else{
						        $str.="assign $m2_wires_prefix"."$m2_filter_pattern_p$flag_m2$m2_filter_pattern_s=$m1_wires_prefix"."$m1_filter_pattern_p$flag_m1$m1_filter_pattern_s;//$line//$flag_m1//$line2\n";
					        }
					        $found=1;
					        last;
					}
				}
			}
			if($found==0){
				$str.="//WARNING:Connection Not Found:$line//$flag_m1\n";
				if($direction==0){
					#$str.="assign $m1_wires_prefix"."$m1_filter_pattern_p$flag_m1$m1_filter_pattern_s='b0;//Tie this signal to 0\n\n";
				}	
				#$str.="\n";
			}
		}
	}
	return $str;
}
sub gen_instance{
	my $str="";
	my $module_name=$_[0];
	my $id=$_[1];
	
	my $ports=$_[2];
	my $paras=$_[3];
	
	if($id>=0){
		$str.="\n//instantiate  u_$module_name\_$id---------------------------\n";
		$str.="$module_name $paras u_$module_name\_$id(\n";
	}
	else{
		$str.="\n//instantiate  u_$module_name---------------------------\n";
		$str.="$module_name $paras u_$module_name(\n";
	}
	
	@lines=split("\n",$ports);
	my $line_num=scalar @lines;	
	for($i=0;$i<$line_num;$i++){
		if($lines[$i]=~/(\w+)\s*;/){
			if($i!=$line_num-1){
				if($id>=0){
					$str.="  .$1($module_name\_$id\_$1),//".$lines[$i]."\n" ;
				}
				else{
					$str.="  .$1($module_name\_$1),//".$lines[$i]."\n" ;
				}
			}
			else{
				if($id>=0){
					$str.="  .$1($module_name\_$id\_$1)//".$lines[$i]."\n" ;
				}
				else{
					$str.="  .$1($module_name\_$1)//".$lines[$i]."\n" ;
				}
			}
		}
	}
	$str.=");\n";

	return $str;
}
sub get_ports_of_submodule{
        my $module_name=&trim($_[0]);
        my $submodule_num=@{$main::top_design{background}{submodule}{name}};
        my $ports="";
        
        for(my $i=0;$i<$submodule_num;$i++){
                        if($module_name eq $main::top_design{background}{submodule}{name}[$i]){
                                $ports=$main::top_design{background}{submodule}{ports}[$i];
                                last;
                        }
                }
        return $ports;
}
sub get_inst_num_of_submodule{
        my $module_name=&trim($_[0]);
        my $inst_num=0;
        
        if($main::top_design{instance}{module}){
                my $inst_submodule_num=@{ $main::top_design{instance}{module} };
                
                for(my $i=0;$i<$inst_submodule_num;$i++){
                        if($module_name eq $main::top_design{instance}{module}[$i]){
                                $inst_num=$main::top_design{instance}{num}[$i];
                                last;
                        }
                }
        }
        return $inst_num;
}
sub get_id_of_submodule{
        my $module_name=&trim($_[0]);
        my $submodule_num=@{$main::top_design{background}{submodule}{name}};
        my $id;
        for(my $i=0;$i<$submodule_num;$i++){
                        if($module_name eq $main::top_design{background}{submodule}{name}[$i]){
                                $id=$i;
                                last;
                        }
        }
        return $id;
}
sub gen_wire_declaration{
	my $module_name=$_[0];
	my $id=$_[1];
	my $ports=$_[2];
	my $str="";
	
	if($id>=0){
		$str.="\n//wires of $module_name\_$id---------------------------\n";
	}
	else{
		$str.="\n//wires of $module_name---------------------------\n";
	}
	foreach my $line  (split("\n",$ports)){
				my $line_ori=$line;
				$line=&rm_redundancy_spaces($line);
				$line=~s/^\s*input/wire /g;
				$line=~s/^\s*output/wire /g;
				$line=~s/^\s*inout/wire /g;
				$line=~s/\s+reg\s/ /g;
				$line=~s/\s+reg\[/ \[/g;
				if($id>=0){
					$line=~s/(\w+)\s*;/$module_name\_$id\_$1;/g 
				}
				else{
					$line=~s/(\w+)\s*;/$module_name\_$1;/g 
				}
				$str.=$line."//$line_ori\n";
	}
	return $str;
}
sub extract_ports{
        my $src_file=$_[0];
        my $type=$_[1];
        $str="";
        
        if($type eq "file"){
                open FILE ,"< $src_file" or die "Cannot open file $src_file : $!";
                while(<FILE>){
                        chomp;
                        my $line=$_;
                        $line=~s/\/\/.*//;
                        $line=~s/\/\*.*//;
                        if($line=~/^\s*(input|output|inout)\s*/){
                                $line=~s/\)\s*[;,]?\z//;
                                $line=~s/,//;
                                $line=~s/;//;
                                $line=~s/ wire / /;
                                $line=~s/ wire\[/ \[/;
                                $line=~s/ reg / /;
                                $line=~s/ reg\[/ \[/;
                                $line=&trim($line);
                                $line=$line.";\n";
                                $str.=$line;
                        }
                }
                close FILE;
                $main::log_str.="Extracting Ports: $src_file----------------------\n\n$str\n\n";
        }
        elsif($type eq "str"){
                my $src_str=$src_file;
                my @lines=split("\n",$src_str);
                foreach my $line (@lines){
                        $line=~s/\/\/.*//;
                        $line=~s/\/\*.*//;
                        if($line=~/^\s*(input|output|inout)\s*/){
                                $line=~s/,//;
                                $line=~s/;//;
                                $line=~s/ wire / /;
                                $line=~s/ wire\[/ \[/;
                                $line=~s/ reg / /;
                                $line=~s/ reg\[/ \[/;
                                $line=&trim($line);
                                $line=$line.";\n";
                                $str.=$line;
                        }
                }
                $main::log_str.="Extracting Ports: from str----------------------\n\n$str\n\n";
        }
        return $str;
}

sub rm_redundancy_spaces{
	my $str=$_[0];
	$str=~s/\t/ /g;
	$str=~s/\s+/ /g;
	return $str;
}
sub trim{
        my $str=$_[0];
	$str=~s/^\s+//;
	$str=~s/\s+$//;
	return $str;
}
sub get_paras{
	my $die_msg="Unrecoginized argument.\nUsage:module_composer.pl -i [module.xml] -output_dir(optional) [OUTPUT_DIR]\n";
	$main::output_dir="./";
	while($#ARGV>-1){
		if($ARGV[0]=~/-i/i){
			if($ARGV[1]!~/^-/){
				$main::infile=$ARGV[1];
				shift @ARGV;
			}
			shift @ARGV;
		}
		elsif($ARGV[0]=~/-output_dir/i){
			if($ARGV[1]!~/^-/){
				$main::output_dir=$ARGV[1];
				shift @ARGV;
			}
			shift @ARGV;
		}
		else{
			last;#die $die_msg;
		}
	}
	if(!$main::infile){
		die $die_msg;
	}
}


