#!/usr/bin/perl -w
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


use strict;
use warnings;
use Cwd;
use Cwd 'chdir';
use Getopt::Long;
use File::Basename;

use Term::ReadLine;

# TODO: accept STDIN as well as inFile. 

my $usage = "
This script will tile/untile an input text file with each sample on a newline.
use it thus:
matrix_mult_datafile_partition.pl -f data/inputA.txt -tileRow 2 -tileCol 4 -r 16 -c 32
    The above will output a data/inputAtiled.txt file assuming that 16x32 row major matrix input. Will tile with 2x4 pattern.
options:
    -f|--file|--inFile=s       => input filepath containing input matrix of size r_c. mandatory.
    -m|--mtile|--tileRow=i     => tileRows (M) dimension for AIE API mmult scheme
    -n|--ntile|--tileCol=i     => tileCols (N) dimension for AIE API mmult scheme
    -r|--inRow=i               => Actual number of rows for InMatrix
    -c|--inCol=i               => Actual number of cols for InMatrix
    -p|--partition|--cascLen=i => Number of partitions / Cascade length for InMatrix
    [--splitRows]              => Optional. Specify if input data is to be partitioned over rows. Default behaviour assumes split over columns. 
    [--isTiled]                => Optional. Specify if input data is already tiled. Default behaviour assumes it is not tiled. 
    [-o|--outFile=s]           => Optional. output filepath (default is inFilePath/inFileName_<casc_index>.<inFileExt>)
    [--tileInPlace]            => Optional. Specificy if tiling should happen in-place or be given a suffix of _tiled or _untiled. ,
    [--colMajor]               => Optional. Specifies that the InMatrx is stored colMajor. Output will be tiled&rowMajor.
    [--untile]                 => Optional. the input matrix is un-tiled. Works with other options, ie if colMajor present output will be stored ColumnMajor
    [-h|--help]                => Optional. prints this usage
    [-v|--verbose]             => Optional. additional logging

";


my $inFile = "";
my $outFile = "";
my $inRow = "";
my $inCol = "";
my $cascLen = "";
my $splitRows  = 0;
my $verbose = 0;
my $untile  = 0;
my $inplace = 0;
my $isTiled  = 0;
my $colMajor = 0;
my $help = 0;
my $T_DATA_A = "";
my $T_DATA_B = "";
GetOptions (
            "f|file|inFile=s"       => \$inFile,  # string
            "o|outFile=s"           => \$outFile,  # string
            "r|inRow=i"             => \$inRow,
            "c|inCol=i"             => \$inCol,
            "p|partition|cascLen=i" => \$cascLen,
            "splitRows"             => \$splitRows,
            "untile"                => \$untile,
            "isTiled=i"             => \$isTiled,
            "--tileInPlace"         => \$inplace,
            "colMajor=i"            => \$colMajor,
            "T_DATA_A=s"            => \$T_DATA_A,
            "T_DATA_B=s"            => \$T_DATA_B,
            "h|help"                => \$help,
            "v|verbose"             => \$verbose) # flag
or die("Error in command line arguments\n");

if ( $help ) { 
    die "$usage";
}

# TODO: command line arguments for tile / untile and inplace / filename_tiled.txt

# Handle mandatory arguments
if ( $inFile eq "" ) { 
    die "ERROR: Provide command line argument for inFile. Use -h for usage. ";
}

if ( $T_DATA_A eq "" ) { 
    die "ERROR: Provide command line argument for T_DATA_A. Use -h for usage. ";
}
if ( $T_DATA_B eq "" ) { 
    die "ERROR: Provide command line argument for T_DATA_B. Use -h for usage. ";
}

if ( $inRow eq "" ) { 
    die "ERROR: Provide command line argument for inRow. Use -h for usage. ";
}

if ( $inCol eq "" ) { 
    die "ERROR: Provide command line argument for inCol. Use -h for usage. ";
}

# Handle verbose
if ( $verbose ) { 
    if ( $inFile ne "" ) { 
        print "inFile is $inFile. \n";
    }
    if ( $inRow ne "" ) { 
        print "inRow is $inRow. \n";
    }
    if ( $inCol ne "" ) { 
        print "inCol is $inCol. \n";
    }

    if  ($colMajor) { 
        print "colMajor is enabled\n";
    }
    if  ($untile) { 
        print "untile is enabled\n";
    }
    if  ($help) { 
        print "help is enabled\n";
    }
    if  ($verbose) { 
        print "verbose is enabled\n";
    }
    if ($outFile ne "" ) { 
        print "outFile is $outFile. \n";
    }
}


# Default to sensible value
my ${DIM_A_TILE}  = 4;
my ${DIM_AB_TILE} = 4;
my ${DIM_B_TILE}  = 2;

# CAUTION: Maintenance hazard - these definitions are duplicated in metadata and matrix_mult.hpp
if ( ${T_DATA_A} eq "int16" ) {
	if ( ${T_DATA_B} eq "int16" ) {
		${DIM_A_TILE}  = 4;
		${DIM_AB_TILE} = 4;
		${DIM_B_TILE}  = 4;
	}
	
	if ( ${T_DATA_B} eq "cint16" ) {
		${DIM_A_TILE}  = 4;
		${DIM_AB_TILE} = 2;
		${DIM_B_TILE}  = 2;
	}

	if ( ${T_DATA_B} eq "int32" ) {
		${DIM_A_TILE}  = 4;
		${DIM_AB_TILE} = 2;
		${DIM_B_TILE}  = 2;
	}
	
	if ( ${T_DATA_B} eq "cint32" ) {
		${DIM_A_TILE}  = 2;
		${DIM_AB_TILE} = 4;
		${DIM_B_TILE}  = 2;
	}

}

if ( ${T_DATA_A} eq "cint16" ) {
	if ( ${T_DATA_B} eq "int16" ) {
		${DIM_A_TILE}  = 4;
		${DIM_AB_TILE} = 4;
		${DIM_B_TILE}  = 2;
	}
	
	if ( ${T_DATA_B} eq "cint16" ) {
		${DIM_A_TILE}  = 4;
		${DIM_AB_TILE} = 4;
		${DIM_B_TILE}  = 2;
	}

	if ( ${T_DATA_B} eq "int32" ) {
		${DIM_A_TILE}  = 4;
		${DIM_AB_TILE} = 4;
		${DIM_B_TILE}  = 2;
	}
	
	if ( ${T_DATA_B} eq "cint32" ) {
		${DIM_A_TILE}  = 2;
		${DIM_AB_TILE} = 2;
		${DIM_B_TILE}  = 2;
	}

}

if ( ${T_DATA_A} eq "int32" ) {
	if ( ${T_DATA_B} eq "int16" ) {
		${DIM_A_TILE}  = 4;
		${DIM_AB_TILE} = 4;
		${DIM_B_TILE}  = 2;
	}
	
	if ( ${T_DATA_B} eq "cint16" ) {
		${DIM_A_TILE}  = 4;
		${DIM_AB_TILE} = 4;
		${DIM_B_TILE}  = 2;
	}

	if ( ${T_DATA_B} eq "int32" ) {
		${DIM_A_TILE}  = 4;
		${DIM_AB_TILE} = 4;
		${DIM_B_TILE}  = 2;
	}
	
	if ( ${T_DATA_B} eq "cint32" ) {
		${DIM_A_TILE}  = 2;
		${DIM_AB_TILE} = 2;
		${DIM_B_TILE}  = 2;
	}

}


if ( ${T_DATA_A} eq "cint32" ) {
	if ( ${T_DATA_B} eq "int16" ) {
		${DIM_A_TILE}  = 2;
		${DIM_AB_TILE} = 4;
		${DIM_B_TILE}  = 2;
	}
	
	if ( ${T_DATA_B} eq "cint16" ) {
		${DIM_A_TILE}  = 2;
		${DIM_AB_TILE} = 2;
		${DIM_B_TILE}  = 2;
	}

	if ( ${T_DATA_B} eq "int32" ) {
		${DIM_A_TILE}  = 2;
		${DIM_AB_TILE} = 2;
		${DIM_B_TILE}  = 2;
	}
	
	if ( ${T_DATA_B} eq "cint32" ) {
		${DIM_A_TILE}  = 2;
		${DIM_AB_TILE} = 2;
		${DIM_B_TILE}  = 2;
	}

}


if ( ${T_DATA_A} eq "float" ) {
	if ( ${T_DATA_B} eq "float" ) {
		${DIM_A_TILE}  = 4;
		${DIM_AB_TILE} = 4;
		${DIM_B_TILE}  = 2;
	}
	
	if ( ${T_DATA_B} eq "cfloat" ) {
		${DIM_A_TILE}  = 2;
		${DIM_AB_TILE} = 4;
		${DIM_B_TILE}  = 2;
	}

}

if ( ${T_DATA_A} eq "cfloat" ) {
	if ( ${T_DATA_B} eq "float" ) {
		${DIM_A_TILE}  = 2;
		${DIM_AB_TILE} = 4;
		${DIM_B_TILE}  = 2;
	}
	
	if ( ${T_DATA_B} eq "cfloat" ) {
		${DIM_A_TILE}  = 4;
		${DIM_AB_TILE} = 2;
		${DIM_B_TILE}  = 2;
	}

}


my $tileRow = "";
my $tileCol = "";
my $dataType = "";
# default tiler gets provided dimensions, if cascade, we divide AB by casc len. 
my $tileInRow = $inRow;
my $tileInCol = $inCol;
if ( $cascLen eq "" ) { 
    # using output
    $tileRow = $DIM_A_TILE;
    $tileCol = $DIM_B_TILE;
    # we only use output type to determine if int16, which only happens when both types are int16. 
    $dataType = $T_DATA_A;

} elsif ( $splitRows ) {
    # using B
    $tileRow = $DIM_AB_TILE;
    $tileCol = $DIM_B_TILE;
    $dataType = $T_DATA_B;
    $tileInRow = ( $inRow / $cascLen );

} else { 
    # using A
    $tileRow = $DIM_A_TILE;
    $tileCol = $DIM_AB_TILE;
    $dataType = $T_DATA_A;
    $tileInCol = ( $inCol / $cascLen );
}


# get component parts of input/output filenames
(my $inFileName, my $inFileDir, my $inFileExt) = fileparse($inFile, '\..*');

my $outFileName;my $outFileDir;my $outFileExt;my $outFileTempName;
if ($outFile ne "" ) { 
    ($outFileTempName, $outFileDir, $outFileExt) = fileparse($outFile, '\..*');
    $outFileName = "${outFileTempName}_";
} else { 
    $outFileName = "${inFileName}_";
    $outFileDir = $inFileDir;
    $outFileExt = $inFileExt;
    #print "$outFileName  : $outFileDir  :  $outFileExt \n" ; 
}


print "Reading $inFile. \n";
print "isTiled is $isTiled\n";



my @resOutFiles;

my @inText;
if ( $cascLen eq "" ) { 
    # in this case, output is stil tiled and needs detiling. 
    if ( ! $isTiled ) { 
        if ($dataType eq "int16") {
            rename($inFile, $inFile . '.beforeOutputInt16LinePerSample');
            open(IN, '<' . $inFile . '.beforeOutputInt16LinePerSample') or die $!;
            open(OUT, '>' . $inFile) or die $!;

            while (<IN>) {
                s/\s+(-?[0-9]+)\s?/\n$1/g ;
                print OUT $_;
            }
            close(OUT)
                or die "couldn't close OUT";
            close(IN)
                or die "couldn't close IN";
        }

        tile_matrix($inFile);
    }

} else { 
    
    if ($dataType eq "int16") {
        rename($inFile, $inFile . '.beforeInt16LinePerSample');
        open(IN, '<' . $inFile . '.beforeInt16LinePerSample') or die $!;
        open(OUT, '>' . $inFile) or die $!;
        while (<IN>) {
            s/\s(-?[0-9]+)//g ;
            print OUT $_;
        }
        close(OUT)
            or die "couldn't close OUT";
        close(IN)
            or die "couldn't close IN";
    }
    open(my $inFileh, "<" , $inFile)
        or die "Can't open < $inFile";
    
    while(<$inFileh>) { 
        chomp;
        #print "$_ \n";
        #if ( $dataType eq "int16" ) { 
        #    $_ =~ s/\s(-?[0-9]+)//g;
        #}
        push @inText, $_;
    }


    close($inFileh)
        or die "couldn't close inFileh $inFileh";
    
    if ($dataType eq "int16") {
        int16_twoSamplesPerLine($inFile);
    }
    partition_matrix();

    print "\n Writing to :\n";
    print join(", ", @resOutFiles);
    print "\n";
    if ( ! $isTiled ) {
        for my $fileForTile (@resOutFiles) { 
            tile_matrix($fileForTile);
        } 
    }
}

sub partition_matrix { 

    
    
    my @duplicateText = @inText;

    my $colsPerCasc;
    my $rowsPerCasc;
    if ( $splitRows ) {
        $rowsPerCasc = $inRow / $cascLen ; 
        $colsPerCasc = $inCol;
    } else { 
        $colsPerCasc = $inCol / $cascLen ; 
        $rowsPerCasc = $inRow;
    }
    my @cascades = (0...($cascLen - 1));
    my @columns = (0...($colsPerCasc - 1));
    my @rows = (0...($rowsPerCasc - 1));
    my @batches = (0...((($#inText + 1) / ( $inRow * $inCol ))) - 1);

    my $colElementDist;
    my $colElementDistCasc;
    my $rowElementDist;
    my $rowElementDistCasc;
    if ( $colMajor ) {
        $rowElementDist = 1;
        $rowElementDistCasc = 1;
        $colElementDist = $inRow;
        $colElementDistCasc = $rowsPerCasc;
    } else { 
        $rowElementDist = $inCol;
        $rowElementDistCasc = $colsPerCasc;
        $colElementDist = 1;
        $colElementDistCasc = 1;
    }

    if ( $verbose ) { 
        print "columns: ",join(", ", @columns), "\n";
        print "rows: ",join(", ", @rows), "\n";
        print "batches: ",join(", ", @batches), "\n";
    }

    # create resultant outputfile names and handlers.
    my @outFileh;
    #my @resOutFiles;
    print "writing output files\n";
    for my $file (@cascades) {
        $resOutFiles[$file] = "${outFileDir}${outFileName}${file}${outFileExt}";
        print "$resOutFiles[$file] \n";
        open($outFileh[$file], ">", $resOutFiles[$file])
           or die "cannot open $resOutFiles[$file]: $!";
    }

    
    my $currentFile;
    my $inTextIndex;
    my $colIndex;
    my $rowIndex;
    my @outputArray;
    for my $batch (@batches){
        if ( $colMajor ) { 
            for my $col (@columns){ 
                for my $row (@rows){
                    for my $cascI ((0...($cascLen - 1))) {
                        $currentFile = $outFileh[$cascI];
                        if ( $splitRows ) { 
                            $colIndex = $col;
                            $rowIndex = ( $row + ( $cascI * $rowsPerCasc ) );
                        } else {
                            $colIndex = ( $col + ( $cascI * $colsPerCasc ) );
                            $rowIndex = $row;
                        }
                        $inTextIndex = (($batch * $inRow * $inCol) + ($rowIndex * $rowElementDist) + ( $colIndex * $colElementDist ));
                        print $currentFile "$inText[$inTextIndex]\n";
                        if ( $verbose )  { 
                            print "inText[$inTextIndex] = $inText[$inTextIndex]\n";
                        }
                    }
                }
            }

        } else { 
            for my $row (@rows){
                for my $col (@columns){ 
                    for my $cascI ((0...($cascLen - 1))) {
                        $currentFile = $outFileh[$cascI];
                        if ( $splitRows ) { 
                            $colIndex = $col;
                            $rowIndex = ( $row + ( $cascI * $rowsPerCasc ) );
                        } else {
                            $colIndex = ( $col + ( $cascI * $colsPerCasc ) );
                            $rowIndex = $row;
                        }
                        $inTextIndex = (($batch * $inRow * $inCol) + ($rowIndex * $rowElementDist) + ( $colIndex * $colElementDist ));
                        print $currentFile "$inText[$inTextIndex]\n";
                        if ( $verbose )  { 
                            print "inText[$inTextIndex] = $inText[$inTextIndex]\n";
                        }
                    }
                }
            }
        }
    }

    # Finally write out resultant data to each file. 
    for my $file (@cascades) {
        for my $i (@{$outputArray[$file]}) { 
            print "$i \n";
        }
        close($outFileh[$file])
            or die "couldn't close outFileh $outFileh[$file]: $!";
        
        if ( $dataType eq "int16" && $isTiled ) { 
            print "2 samples per line for partitioning int16\n";
            int16_twoSamplesPerLine($resOutFiles[$file]);
        }
    }

    
    print "Finished writing @resOutFiles .\nEnd of partitioning.\n";

}

sub tile_matrix { 
    my ($fileForTile) = @_ ; 
    (my $fileForTileName, my $fileForTileDir, my $fileForTileExt) = fileparse($fileForTile, '\..*');

    my $outTileFileName;my $outTileFileDir;my $outTileFileExt;
    if ($inplace) {

        $outTileFileName = $fileForTileName;
        $outTileFileDir = $fileForTileDir;
        $outTileFileExt = $fileForTileExt;

    } else {
        
        if ($outFile ne "" ) { 
            ($outTileFileName, $outTileFileDir, $outTileFileExt) = fileparse($outFile, '\..*');
        } else {
            my $un = "";
            if ($untile) { $un = "un"; }
            $outTileFileName = "${un}tiled_${fileForTileName}";
            $outTileFileDir = $fileForTileDir;
            $outTileFileExt = $fileForTileExt;
        }
    }
    #print "$outTileFileName  : $outTileFileDir  :  $outTileFileExt \n" ; 
    my $resOutTileFile = "${outTileFileDir}${outTileFileName}${outTileFileExt}";

    print "outTileFile is $resOutTileFile\n";

    

    #print "$inFileName : $inFileDir  : $inFileExt \n";
    print "Reading $fileForTile. \n";

    open(my $fileForTileh, "<" , $fileForTile)
        or die "Can't open < $fileForTile";

    #my $line = readline($fileForTileh);
    #print($line);
    #$line = readline($fileForTileh);
    #print($line);
    my @inTileText;
    while(<$fileForTileh>) { 
        chomp;
        push @inTileText, $_;
    }

    close($fileForTileh)
        or die "couldn't close fileForTileh $fileForTileh";
    
    if ($inplace) {
        rename($fileForTile, $fileForTile . '.beforeTile'); # create a backup file
    }
    print "Finished reading file\n";

    
    my @duplicateText = @inTileText;
    my @transText = @inTileText;
    # fill with dummy data basically
    my @indices = @inTileText;
    my @transIndices = @inTileText;
    #$((((($i-1)/($AB*$K))*$K + (($i-1)%$K))*$AB + ((($i-1)/$K) % $AB) +1 ))
    #tileInRow
    #tileInCol
    #
    #tileRow
    #tileCol

    #res=$(( (( (($i-1)/($AB*$M))*($AB*$M) + ((($i-1)/$N) % $M)) * $AB) +  (($i-1) % $N) + ((((($i-1)/($N*$M)) * $N) %  $AB)  + 1 ) ))
    #open(my $outFileh, ">" , $resOutFile)
    #    or die "Can't open > $resOutFile";

    print "Shuffling indicies\n";
    my @iIter = (0..$#inTileText);
    for my $i (@iIter){ 
        my $newIndex;
        my $transposeIndex;
        {
            use integer;
            my $colI = ( $untile ) ? ($i % $tileInRow) : ($i % $tileInCol);
            my $colIncr = ( $untile ) ? $tileInCol: $tileInRow;
            my $rowI = ( $untile ) ? (($i / $tileInRow) % $tileInCol) : (($i / $tileInCol) % $tileInRow);
            my $rowIncr = 1;
            my $batchI = ($i/($tileInCol*$tileInRow)); #unchanged
            my $batchIncr = $tileInCol*$tileInRow; #unchanged

            $transposeIndex = ($colI * $colIncr) + ( $rowI * $rowIncr) + ($batchI * $batchIncr);
            #print "transposeIndex: ($colI * $colIncr) + ($rowI * $rowIncr) + ($batchI * $batchIncr) = $transposeIndex\n ";

            # fine-grained within a chunk index
            my $colInTileI =  ($i % $tileCol); 
            my $colInTileIncr = 1; 

            # which chunk of N samples within tile row
            my $rowInTileI = (( $i/$tileCol ) % $tileRow); 
            my $rowInTileIncr = $tileInCol; # grab next row for each tileRow. 
            
            my $tileIndex = ($i/($tileRow*$tileCol));
            my $tileIncr = $tileCol; # advance further down the row ;

            # Which tile in row of tiles
            my $tileWithinRowOffset = ( $tileIndex * $tileIncr ) % $tileInCol;

             # Coarse grain - increments of row of tile
            my $rowOfTileIndex = ($i/( $tileInCol * $tileRow ));
            my $rowOfTileIncr = ( $tileInCol * $tileRow );


            # force everything to be integer arithmetic
                                        
            $newIndex = ($rowOfTileIndex*$rowOfTileIncr) +  $tileWithinRowOffset + ($rowInTileI*$rowInTileIncr) + ($colInTileI * $colInTileIncr);

            #print "newIndex: ($rowOfTileIndex*$rowOfTileIncr) + $tileWithinRowOffset + ($rowInTileI*$rowInTileIncr) + ($colInTileI * $colInTileIncr) = $newIndex\n";
        }
        #print "$i ($newIndex) => $transposeIndex \n";
        #print "$transposeIndex $transposeIndex\n";

        if ($untile) {
            $indices[$newIndex] = $i;
        } else { 
            $indices[$i] = $newIndex;
        }
        #if ($colMajor) {
            $transIndices[$i] = $transposeIndex;
        #}
        #print (int $i/( $tileInCol * $tileCol )); 
        #print "$newIndex \n";
    }


    print "Writing $resOutTileFile. \n";
    open(my $outFileh, ">" , $resOutTileFile)
        or die "Can't open > $resOutTileFile";
    #my @iIter = (0..$#inText/8);
    for my $i (@iIter){ 
        #if ($colMajor) {
        #    $duplicateText[$i] = $inText[$transIndices[$indices[$i]]];
        #} else {
            $duplicateText[$i] = "$inTileText[$indices[$i]]";
        #}
    }
    for my $i (@iIter){ 
        if ($untile) { 
            $transText[$i] = "$duplicateText[$transIndices[$i]]";
        } else { 
            $transText[$i] = "$inTileText[$transIndices[$indices[$i]]]";
        }
        if ($colMajor) {
            print $outFileh "$transText[$i] \n";
        } else {
            print $outFileh "$duplicateText[$i] \n";
        }
        if ($verbose) {
            print "$i = $indices[$i] = $transIndices[$i]      \t \t ";
            print "$inTileText[$i] => $duplicateText[$i] => $transText[$i] \n";
        }
        #print "$i $i\n";
    }

    close($outFileh)
        or die "couldn't close outFileh $outFileh";

    if ( $dataType eq "int16" ) { 
        print "2 samples per line for tiling int16\n";
        int16_twoSamplesPerLine($resOutTileFile);
    }
    print "Finished writing $resOutTileFile .\nEnd of tiling.\n";
}

sub int16_twoSamplesPerLine { 
    my ($fileToParse) = @_ ; 
    rename($fileToParse, $fileToParse . '.beforeInt16EditResult');
    open(IN, '<' . $fileToParse . '.beforeInt16EditResult') or die $!;
    open(OUT, '>' . $fileToParse) or die $!;
    my $counter = 1;
    while (<IN>) {
        #print "in is ${_}ok and is num $counter\n";
        s/\s?\n/ /g if $counter % 2;
        print OUT $_;
        #print "out is ${_}ok\n";
        $counter = ( $counter + 1 );
    }
    close(OUT)
        or die "couldn't close OUT";


}