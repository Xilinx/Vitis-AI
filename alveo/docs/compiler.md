# Compiler Optimization(s) for DPU-v1

Would you like to understand the meaning of the compiler flags? I
would, especially if compilation fails, for any reason, you would like
to change how code is generated. This is a short overview of the
python command line interface, we try to keep the python class
interface consistent. Please, let us know if there is any unintended
difference.

## Replication (V3)

Our Convolution-Engine can achieve peak performance if its input
tensor has a number of channels multiple of 96. Any remainder and we
are leaving performance on the table. Replication is a hardware
functionality that the compiler can activate in order to take back
some of the performance left by the mod(channels,96)>0.  The compiler
provides by default a safe activation of replication and in the future
it will deploy more aggressive ones. There are corner cases where
"replication" may require more memory especially small-channel
tensors, spilling the computation into DDR. You may turn off
replication by the compiler flag `--noreplication`.

If you are a hardware expert, and your name starts with a "V", there
is a way to override the compiler replication altogether.

You may see `--customreplication "First Layer"`. This is an intermediary
solution for the first layer where replication is set differently.  As
for when you are reading this, this optimization request could be
redundant. The compiler will handle differently and optimally the
first layer when it is stored into DDR.

The compiler will provide feedback of the replication process during the parsing.

## Tile selection and Tiling 

Basically, we have a two-level memory hierarchy: DDR large and slow,
AM small and fast. When the computation does not fit AM, we tile
it. The hardware has support for tiling and the compiler provides
basic information to the hardware (i.e., tile size). The compiler has
a default optimization process for the selection of the tiles: it
tries to minimize the overlap of input data between iterations, to
have the largest "width" of the input tensor (exploiting spatial
locality and memory throughput). In its optimization arsenal, the
compiler can take direct suggestions (i.e., V3 experts) or use an
internal performance model to choose the tile size with best overall
performance (experimental). Tiling cannot be turned off for obvious
reason (no computation).

## Pipelineconvmaxpool

The systolic array, which computes the convolution, can pipeline the
MaxPool operation. So the compiler can fuse the convolution + maxpool
into a single instruction. The advantage of the pipeline is: first,
improved efficiency (saving a write and a read into AM) and, second,
the deletion of the temporary tensor in between convolution and
pooling, saving space and improve memory pressure. This optimization
is turned on by `--pipelineconvmaxpool`. In general, this is always
beneficial and please keep it on.

## Pooling Around

It is a pun. To exploit the possibility to pipeline more convolution
with pool layers, `--poolingaround` look for maxpool operations outside
an inception and bring it into if the concatenation is build by all
convolutions.In combination with the optimization above we improve
efficiency and very often memory space use.

## Parallel Read

The AM memory has two read ports and two write ports. In principle, we
could read two operands in parallel if such operands are stored in AM
appropriately. The ElementWise operation takes two operands and add
them (hopefully it writes the results in one of them). The compiler
address this optimization by an heuristic or suggestion that is passed
to the memory allocation: `--parallelread "['tops','bottom']"`

The memory allocation phase of the compiler is when, off line, we
allocate space for the vectors in AM. To exploit read parallelism, you
are passing two opposite heuristics to the compiler for the two
different operands: very briefly, "tops" means that one operand should
be allocated (if possible) to the top of the memory AM and, you
guessed it, the other to the "bottom". This a suggestion that is
applied to all EltWise operations. The compiler during code generation
will let you know if it was able to parallelize the reads. If it fails
the code is still valid but not performant.
```
PARALLEL ELTREAD >
 res3a Wait(Wait_Download=1, Wait_Upload=1, Wait_Conv=1, Wait_Pool=1, Wait_EW=1, Wait_Upsmpl=1, ParalleRead=1)
  IN 7781376-8232960
  IN 0-451584
NOT PARALLEL ELTREAD >
 res3b Wait(Wait_Download=1, Wait_Upload=1, Wait_Conv=1, Wait_Pool=1, Wait_EW=1, Wait_Upsmpl=1, ParalleRead=0) AM Partition [0 3145728 6291456 9437184]
  IN 7781376-8232960
  IN 8985600-9437184
```

## Instruction Parallelism (-aka -P and -parallelismstrategy)

Above we showed how to explore data parallelism in a single
instruction. Here, we explore the compiler phase where we explore
instruction parallelism within inceptions. The compiler spells out
this phase quite verbosely. Once you activate the `-P --parallelism`
flag, the compiler will try to figure out the inceptions where
parallelism can be found.

Convolutions ( and pipeline conv + pool) can be executed in parallel
with Scale, Pool and Element-wise. It is like we have two functional
units and we need to create a valid schedule.

Currently there are 3 algorithms "tfs", "inc", "sn", the last one
should encompass all and they can be set by the `-Pin` or
`--parallelismgraphalgorithm` ... if it is availble. Thus by `-P -Pinc
sn`, we activate the parallel search and solution space. As you can
imagine to exploit instruction parallelism we must explore also data
parallelism.  By using `-Q, –parallelismstrategy`, we can give
suggestions to the memory allocation.

`-PInc  tfs --poolingaround -Q "['tops','bottom']"`  works for googlenetv1

`-PInc  inc --poolingaround -Q "['tops','bottom']"` for resnet

but if it is available `-Pinc sn` works for inception_v3, v4.

The compiler will let you know that parallelism has been found by
using a Clint Eastwood's phrase and then will check if data
parallelism has been found. If parallelism is not found you may try to
swap tops and bottom position in the suggestion box.

## Mix

The hardware allows memory asymmetric computation. The usual style is
DDR-to-DDR (tiled) and AM-to-AM, that is the inputs and the outputs
reside in the same memory level. These are symmetric computations. For
memory asymmetric computation, we mean DDR-to-AM and
AM-to-DDR. Convolution is the only instructions that allows all
styles. If you activate `-mix –mixmemorystrategy`, you will have the
opportunity to use them. The main advantage is the reduction of data
movements: communications are interleaved with computations.

## Identity Convolution or Identity Scale insertion

Our hardware achieve is peak performance because of a few
assumptions. These may be in contrast with some network designs and
their quantizations. The compiler may apply a few network
modifications: a scale operation can be added in order to to adjust
quantization differences or a convolution can be added to allows
better data alignment. These processes cannot be turned off because
the network without them will have very poor accuracy and even
incorrect results.
