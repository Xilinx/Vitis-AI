# Copyright 2021 Xilinx Inc.
#
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

# Implementation adapted from AlphaNet: https://github.com/facebookresearch/AlphaNet
# Attribution-NonCommercial 4.0 International

# =======================================================================

# Creative Commons Corporation ("Creative Commons") is not a law firm and
# does not provide legal services or legal advice. Distribution of
# Creative Commons public licenses does not create a lawyer-client or
# other relationship. Creative Commons makes its licenses and related
# information available on an "as-is" basis. Creative Commons gives no
# warranties regarding its licenses, any material licensed under their
# terms and conditions, or any related information. Creative Commons
# disclaims all liability for damages resulting from their use to the
# fullest extent possible.

# Using Creative Commons Public Licenses

# Creative Commons public licenses provide a standard set of terms and
# conditions that creators and other rights holders may use to share
# original works of authorship and other material subject to copyright
# and certain other rights specified in the public license below. The
# following considerations are for informational purposes only, are not
# exhaustive, and do not form part of our licenses.

#      Considerations for licensors: Our public licenses are
#      intended for use by those authorized to give the public
#      permission to use material in ways otherwise restricted by
#      copyright and certain other rights. Our licenses are
#      irrevocable. Licensors should read and understand the terms
#      and conditions of the license they choose before applying it.
#      Licensors should also secure all rights necessary before
#      applying our licenses so that the public can reuse the
#      material as expected. Licensors should clearly mark any
#      material not subject to the license. This includes other CC-
#      licensed material, or material used under an exception or
#      limitation to copyright. More considerations for licensors:
#     wiki.creativecommons.org/Considerations_for_licensors

#      Considerations for the public: By using one of our public
#      licenses, a licensor grants the public permission to use the
#      licensed material under specified terms and conditions. If
#      the licensor's permission is not necessary for any reason--for
#      example, because of any applicable exception or limitation to
#      copyright--then that use is not regulated by the license. Our
#      licenses grant only permissions under copyright and certain
#      other rights that a licensor has authority to grant. Use of
#      the licensed material may still be restricted for other
#      reasons, including because others have copyright or other
#      rights in the material. A licensor may make special requests,
#      such as asking that all changes be marked or described.
#      Although not required by our licenses, you are encouraged to
#      respect those requests where reasonable. More_considerations
#      for the public:
#     wiki.creativecommons.org/Considerations_for_licensees

# =======================================================================

# Creative Commons Attribution-NonCommercial 4.0 International Public
# License

# By exercising the Licensed Rights (defined below), You accept and agree
# to be bound by the terms and conditions of this Creative Commons
# Attribution-NonCommercial 4.0 International Public License ("Public
# License"). To the extent this Public License may be interpreted as a
# contract, You are granted the Licensed Rights in consideration of Your
# acceptance of these terms and conditions, and the Licensor grants You
# such rights in consideration of benefits the Licensor receives from
# making the Licensed Material available under these terms and
# conditions.

# Section 1 -- Definitions.

#   a. Adapted Material means material subject to Copyright and Similar
#      Rights that is derived from or based upon the Licensed Material
#      and in which the Licensed Material is translated, altered,
#      arranged, transformed, or otherwise modified in a manner requiring
#      permission under the Copyright and Similar Rights held by the
#      Licensor. For purposes of this Public License, where the Licensed
#      Material is a musical work, performance, or sound recording,
#      Adapted Material is always produced where the Licensed Material is
#      synched in timed relation with a moving image.

#   b. Adapter's License means the license You apply to Your Copyright
#      and Similar Rights in Your contributions to Adapted Material in
#      accordance with the terms and conditions of this Public License.

#   c. Copyright and Similar Rights means copyright and/or similar rights
#      closely related to copyright including, without limitation,
#      performance, broadcast, sound recording, and Sui Generis Database
#      Rights, without regard to how the rights are labeled or
#      categorized. For purposes of this Public License, the rights
#      specified in Section 2(b)(1)-(2) are not Copyright and Similar
#      Rights.
#   d. Effective Technological Measures means those measures that, in the
#      absence of proper authority, may not be circumvented under laws
#      fulfilling obligations under Article 11 of the WIPO Copyright
#      Treaty adopted on December 20, 1996, and/or similar international
#      agreements.

#   e. Exceptions and Limitations means fair use, fair dealing, and/or
#      any other exception or limitation to Copyright and Similar Rights
#      that applies to Your use of the Licensed Material.

#   f. Licensed Material means the artistic or literary work, database,
#      or other material to which the Licensor applied this Public
#      License.

#   g. Licensed Rights means the rights granted to You subject to the
#      terms and conditions of this Public License, which are limited to
#      all Copyright and Similar Rights that apply to Your use of the
#      Licensed Material and that the Licensor has authority to license.

#   h. Licensor means the individual(s) or entity(ies) granting rights
#      under this Public License.

#   i. NonCommercial means not primarily intended for or directed towards
#      commercial advantage or monetary compensation. For purposes of
#      this Public License, the exchange of the Licensed Material for
#      other material subject to Copyright and Similar Rights by digital
#      file-sharing or similar means is NonCommercial provided there is
#      no payment of monetary compensation in connection with the
#      exchange.

#   j. Share means to provide material to the public by any means or
#      process that requires permission under the Licensed Rights, such
#      as reproduction, public display, public performance, distribution,
#      dissemination, communication, or importation, and to make material
#      available to the public including in ways that members of the
#      public may access the material from a place and at a time
#      individually chosen by them.

#   k. Sui Generis Database Rights means rights other than copyright
#      resulting from Directive 96/9/EC of the European Parliament and of
#      the Council of 11 March 1996 on the legal protection of databases,
#      as amended and/or succeeded, as well as other essentially
#      equivalent rights anywhere in the world.

#   l. You means the individual or entity exercising the Licensed Rights
#      under this Public License. Your has a corresponding meaning.

# Section 2 -- Scope.

#   a. License grant.

#        1. Subject to the terms and conditions of this Public License,
#           the Licensor hereby grants You a worldwide, royalty-free,
#           non-sublicensable, non-exclusive, irrevocable license to
#           exercise the Licensed Rights in the Licensed Material to:

#             a. reproduce and Share the Licensed Material, in whole or
#                in part, for NonCommercial purposes only; and

#             b. produce, reproduce, and Share Adapted Material for
#                NonCommercial purposes only.

#        2. Exceptions and Limitations. For the avoidance of doubt, where
#           Exceptions and Limitations apply to Your use, this Public
#           License does not apply, and You do not need to comply with
#           its terms and conditions.

#        3. Term. The term of this Public License is specified in Section
#           6(a).

#        4. Media and formats; technical modifications allowed. The
#           Licensor authorizes You to exercise the Licensed Rights in
#           all media and formats whether now known or hereafter created,
#           and to make technical modifications necessary to do so. The
#           Licensor waives and/or agrees not to assert any right or
#           authority to forbid You from making technical modifications
#           necessary to exercise the Licensed Rights, including
#           technical modifications necessary to circumvent Effective
#           Technological Measures. For purposes of this Public License,
#           simply making modifications authorized by this Section 2(a)
#           (4) never produces Adapted Material.

#        5. Downstream recipients.

#             a. Offer from the Licensor -- Licensed Material. Every
#                recipient of the Licensed Material automatically
#                receives an offer from the Licensor to exercise the
#                Licensed Rights under the terms and conditions of this
#                Public License.

#             b. No downstream restrictions. You may not offer or impose
#                any additional or different terms or conditions on, or
#                apply any Effective Technological Measures to, the
#                Licensed Material if doing so restricts exercise of the
#                Licensed Rights by any recipient of the Licensed
#                Material.

#        6. No endorsement. Nothing in this Public License constitutes or
#           may be construed as permission to assert or imply that You
#           are, or that Your use of the Licensed Material is, connected
#           with, or sponsored, endorsed, or granted official status by,
#           the Licensor or others designated to receive attribution as
#           provided in Section 3(a)(1)(A)(i).

#   b. Other rights.

#        1. Moral rights, such as the right of integrity, are not
#           licensed under this Public License, nor are publicity,
#           privacy, and/or other similar personality rights; however, to
#           the extent possible, the Licensor waives and/or agrees not to
#           assert any such rights held by the Licensor to the limited
#           extent necessary to allow You to exercise the Licensed
#           Rights, but not otherwise.

#        2. Patent and trademark rights are not licensed under this
#           Public License.

#        3. To the extent possible, the Licensor waives any right to
#           collect royalties from You for the exercise of the Licensed
#           Rights, whether directly or through a collecting society
#           under any voluntary or waivable statutory or compulsory
#           licensing scheme. In all other cases the Licensor expressly
#           reserves any right to collect such royalties, including when
#           the Licensed Material is used other than for NonCommercial
#           purposes.

# Section 3 -- License Conditions.

# Your exercise of the Licensed Rights is expressly made subject to the
# following conditions.

#   a. Attribution.

#        1. If You Share the Licensed Material (including in modified
#           form), You must:

#             a. retain the following if it is supplied by the Licensor
#                with the Licensed Material:

#                  i. identification of the creator(s) of the Licensed
#                     Material and any others designated to receive
#                     attribution, in any reasonable manner requested by
#                     the Licensor (including by pseudonym if
#                     designated);

#                 ii. a copyright notice;

#                iii. a notice that refers to this Public License;

#                 iv. a notice that refers to the disclaimer of
#                     warranties;

#                  v. a URI or hyperlink to the Licensed Material to the
#                     extent reasonably practicable;

#             b. indicate if You modified the Licensed Material and
#                retain an indication of any previous modifications; and

#             c. indicate the Licensed Material is licensed under this
#                Public License, and include the text of, or the URI or
#                hyperlink to, this Public License.

#        2. You may satisfy the conditions in Section 3(a)(1) in any
#           reasonable manner based on the medium, means, and context in
#           which You Share the Licensed Material. For example, it may be
#           reasonable to satisfy the conditions by providing a URI or
#           hyperlink to a resource that includes the required
#           information.

#        3. If requested by the Licensor, You must remove any of the
#           information required by Section 3(a)(1)(A) to the extent
#           reasonably practicable.

#        4. If You Share Adapted Material You produce, the Adapter's
#           License You apply must not prevent recipients of the Adapted
#           Material from complying with this Public License.

# Section 4 -- Sui Generis Database Rights.

# Where the Licensed Rights include Sui Generis Database Rights that
# apply to Your use of the Licensed Material:

#   a. for the avoidance of doubt, Section 2(a)(1) grants You the right
#      to extract, reuse, reproduce, and Share all or a substantial
#      portion of the contents of the database for NonCommercial purposes
#      only;

#   b. if You include all or a substantial portion of the database
#      contents in a database in which You have Sui Generis Database
#      Rights, then the database in which You have Sui Generis Database
#      Rights (but not its individual contents) is Adapted Material; and

#   c. You must comply with the conditions in Section 3(a) if You Share
#      all or a substantial portion of the contents of the database.

# For the avoidance of doubt, this Section 4 supplements and does not
# replace Your obligations under this Public License where the Licensed
# Rights include other Copyright and Similar Rights.

# Section 5 -- Disclaimer of Warranties and Limitation of Liability.

#   a. UNLESS OTHERWISE SEPARATELY UNDERTAKEN BY THE LICENSOR, TO THE
#      EXTENT POSSIBLE, THE LICENSOR OFFERS THE LICENSED MATERIAL AS-IS
#      AND AS-AVAILABLE, AND MAKES NO REPRESENTATIONS OR WARRANTIES OF
#      ANY KIND CONCERNING THE LICENSED MATERIAL, WHETHER EXPRESS,
#      IMPLIED, STATUTORY, OR OTHER. THIS INCLUDES, WITHOUT LIMITATION,
#      WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR
#      PURPOSE, NON-INFRINGEMENT, ABSENCE OF LATENT OR OTHER DEFECTS,
#      ACCURACY, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT
#      KNOWN OR DISCOVERABLE. WHERE DISCLAIMERS OF WARRANTIES ARE NOT
#      ALLOWED IN FULL OR IN PART, THIS DISCLAIMER MAY NOT APPLY TO YOU.

#   b. TO THE EXTENT POSSIBLE, IN NO EVENT WILL THE LICENSOR BE LIABLE
#      TO YOU ON ANY LEGAL THEORY (INCLUDING, WITHOUT LIMITATION,
#      NEGLIGENCE) OR OTHERWISE FOR ANY DIRECT, SPECIAL, INDIRECT,
#      INCIDENTAL, CONSEQUENTIAL, PUNITIVE, EXEMPLARY, OR OTHER LOSSES,
#      COSTS, EXPENSES, OR DAMAGES ARISING OUT OF THIS PUBLIC LICENSE OR
#      USE OF THE LICENSED MATERIAL, EVEN IF THE LICENSOR HAS BEEN
#      ADVISED OF THE POSSIBILITY OF SUCH LOSSES, COSTS, EXPENSES, OR
#      DAMAGES. WHERE A LIMITATION OF LIABILITY IS NOT ALLOWED IN FULL OR
#      IN PART, THIS LIMITATION MAY NOT APPLY TO YOU.

#   c. The disclaimer of warranties and limitation of liability provided
#      above shall be interpreted in a manner that, to the extent
#      possible, most closely approximates an absolute disclaimer and
#      waiver of all liability.

# Section 6 -- Term and Termination.

#   a. This Public License applies for the term of the Copyright and
#      Similar Rights licensed here. However, if You fail to comply with
#      this Public License, then Your rights under this Public License
#      terminate automatically.

#   b. Where Your right to use the Licensed Material has terminated under
#      Section 6(a), it reinstates:

#        1. automatically as of the date the violation is cured, provided
#           it is cured within 30 days of Your discovery of the
#           violation; or

#        2. upon express reinstatement by the Licensor.

#      For the avoidance of doubt, this Section 6(b) does not affect any
#      right the Licensor may have to seek remedies for Your violations
#      of this Public License.

#   c. For the avoidance of doubt, the Licensor may also offer the
#      Licensed Material under separate terms or conditions or stop
#      distributing the Licensed Material at any time; however, doing so
#      will not terminate this Public License.

#   d. Sections 1, 5, 6, 7, and 8 survive termination of this Public
#      License.

# Section 7 -- Other Terms and Conditions.

#   a. The Licensor shall not be bound by any additional or different
#      terms or conditions communicated by You unless expressly agreed.

#   b. Any arrangements, understandings, or agreements regarding the
#      Licensed Material not stated herein are separate from and
#      independent of the terms and conditions of this Public License.

# Section 8 -- Interpretation.

#   a. For the avoidance of doubt, this Public License does not, and
#      shall not be interpreted to, reduce, limit, restrict, or impose
#      conditions on any use of the Licensed Material that could lawfully
#      be made without permission under this Public License.

#   b. To the extent possible, if any provision of this Public License is
#      deemed unenforceable, it shall be automatically reformed to the
#      minimum extent necessary to make it enforceable. If the provision
#      cannot be reformed, it shall be severed from this Public License
#      without affecting the enforceability of the remaining terms and
#      conditions.

#   c. No term or condition of this Public License will be waived and no
#      failure to comply consented to unless expressly agreed to by the
#      Licensor.

#   d. Nothing in this Public License constitutes or may be interpreted
#      as a limitation upon, or waiver of, any privileges and immunities
#      that apply to the Licensor or You, including from the legal
#      processes of any jurisdiction or authority.

# =======================================================================

# Creative Commons is not a party to its public
# licenses. Notwithstanding, Creative Commons may elect to apply one of
# its public licenses to material it publishes and in those instances
# will be considered the “Licensor.” The text of the Creative Commons
# public licenses is dedicated to the public domain under the CC0 Public
# Domain Dedication. Except for the limited purpose of indicating that
# material is shared under a Creative Commons public license or as
# otherwise permitted by the Creative Commons policies published at
# creativecommons.org/policies, Creative Commons does not authorize the
# use of the trademark "Creative Commons" or any other trademark or logo
# of Creative Commons without its prior written consent including,
# without limitation, in connection with any unauthorized modifications
# to any of its public licenses or any other arrangements,
# understandings, or agreements concerning use of licensed material. For
# the avoidance of doubt, this paragraph does not form part of the
# public licenses.

# Creative Commons may be contacted at creativecommons.org.

import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.mobilenet import mobilenet_v2

from pytorch_nndct import OFAPruner

IMAGENET_TRAINSET_SIZE = 1281167

parser = argparse.ArgumentParser(description='PyTorch OFA ImageNet Training')

parser.add_argument(
    '--data',
    type=str,
    default='/dataset/imagenet/raw-data',
    help='path to dataset')
parser.add_argument(
    '--pretrained',
    type=str,
    default='mobilenet_v2-b0353104.pth',
    help='Pretrained model path')

# ofa config
parser.add_argument(
    '--channel_divisible', type=int, default=8, help='make channel divisible')
parser.add_argument(
    '--expand_ratio', type=list, default=[0.5, 0.75, 1], help='expand ratio')
parser.add_argument(
    '--excludes',
    type=list,
    default=['features.0.0', 'features.1.conv.0.0', 'features.18.0'],
    help='excludes module')

parser.add_argument(
    '-j',
    '--workers',
    default=32,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=20,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=512,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256), this is the total '
    'batch size of all GPUs on the current node when '
    'using Data Parallel or Distributed Data Parallel')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.01,
    type=float,
    metavar='LR',
    help='initial learning rate',
    dest='lr')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--wd',
    '--weight-decay',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)',
    dest='weight_decay')
parser.add_argument(
    '-p',
    '--print-freq',
    default=100,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of nodes for distributed training')
parser.add_argument(
    '--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument(
    '--dist-url',
    default='tcp://127.0.0.127:23333',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument(
    '--seed', default=20, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument(
    '--multiprocessing-distributed',
    default=True,
    type=bool,
    help='Use multi-processing distributed training to launch '
    'N processes per node, which has N GPUs. This is the '
    'fastest way to use PyTorch for either single node or '
    'multi node data parallel training')
best_acc1 = 0

def sgd_optimizer(model, lr, momentum, weight_decay):
  params = []
  for key, value in model.named_parameters():
    if not value.requires_grad:
      continue
    apply_weight_decay = weight_decay
    apply_lr = lr
    if 'bias' in key or 'bn' in key:
      apply_weight_decay = 0
    if 'bias' in key:
      apply_lr = 2 * lr
    params += [{
        'params': [value],
        'lr': apply_lr,
        'weight_decay': apply_weight_decay
    }]
  optimizer = torch.optim.SGD(params, lr, momentum=momentum)
  return optimizer

def main():
  args = parser.parse_args()

  if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')

  if args.gpu is not None:
    warnings.warn('You have chosen a specific GPU. This will completely '
                  'disable data parallelism.')

  if args.dist_url == "env://" and args.world_size == -1:
    args.world_size = int(os.environ["WORLD_SIZE"])

  args.distributed = args.world_size > 1 or args.multiprocessing_distributed

  ngpus_per_node = torch.cuda.device_count()
  if args.multiprocessing_distributed:
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    args.world_size = ngpus_per_node * args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
  else:
    # Simply call main_worker function
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
  global best_acc1
  args.gpu = gpu

  if args.gpu is not None:
    print("Use GPU: {} for training".format(args.gpu))

  if args.distributed:
    if args.dist_url == "env://" and args.rank == -1:
      args.rank = int(os.environ["RANK"])
    if args.multiprocessing_distributed:
      # For multiprocessing distributed training, rank needs to be the
      # global rank among all the processes
      args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank)

  if not torch.cuda.is_available():
    print('using CPU, this will be slow')
  elif args.distributed:
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    if args.gpu is not None:
      torch.cuda.set_device(args.gpu)

      args.teacher_model = mobilenet_v2()
      args.teacher_model.load_state_dict(torch.load(args.pretrained))

      args.teacher_model.cuda(args.gpu)

      inputs = torch.randn([1, 3, 224, 224], dtype=torch.float32).cuda()

      ofa_pruner = OFAPruner(args.teacher_model, inputs)
      ofa_model = ofa_pruner.ofa_model(args.expand_ratio,
                                       args.channel_divisible, args.excludes)

      model = ofa_model.cuda(args.gpu)

      # When using a single GPU per process and per
      # DistributedDataParallel, we need to divide the batch size
      # ourselves based on the total number of GPUs we have
      args.batch_size = int(args.batch_size / ngpus_per_node)
      args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
      model = torch.nn.parallel.DistributedDataParallel(
          model, device_ids=[args.gpu])

  # define loss function (criterion and kd loss) and optimizer
  criterion = nn.CrossEntropyLoss().cuda(args.gpu)
  soft_criterion = AdaptiveLossSoft().cuda(args.gpu)

  optimizer = sgd_optimizer(model, args.lr, args.momentum, args.weight_decay)

  lr_scheduler = CosineAnnealingLR(
      optimizer=optimizer,
      T_max=args.epochs * IMAGENET_TRAINSET_SIZE // args.batch_size //
      ngpus_per_node)

  # optionally resume from a checkpoint
  if args.resume:
    if os.path.isfile(args.resume):
      print("=> loading checkpoint '{}'".format(args.resume))
      if args.gpu is None:
        checkpoint = torch.load(args.resume)
      else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.resume, map_location=loc)
      args.start_epoch = checkpoint['epoch']
      best_acc1 = checkpoint['best_acc1']
      if args.gpu is not None:
        # best_acc1 may be from a checkpoint from a different GPU
        best_acc1 = best_acc1.to(args.gpu)
      model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      lr_scheduler.load_state_dict(checkpoint['scheduler'])
      print("=> loaded checkpoint '{}' (epoch {})".format(
          args.resume, checkpoint['epoch']))
    else:
      print("=> no checkpoint found at '{}'".format(args.resume))

  cudnn.benchmark = True

  # Data loading code
  traindir = os.path.join(args.data, 'train')
  valdir = os.path.join(args.data, 'val')
  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  train_dataset = datasets.ImageFolder(
      traindir,
      transforms.Compose([
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize,
      ]))

  if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)
  else:
    train_sampler = None

  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=(train_sampler is None),
      num_workers=args.workers,
      pin_memory=True,
      sampler=train_sampler)

  val_dataset = datasets.ImageFolder(
      valdir,
      transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          normalize,
      ]))
  val_loader = torch.utils.data.DataLoader(
      val_dataset,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.workers,
      pin_memory=True)

  for epoch in range(args.start_epoch, args.epochs):
    if args.distributed:
      train_sampler.set_epoch(epoch)

    if epoch == args.start_epoch:
      validate_subnet(train_loader, val_loader, model, ofa_pruner, criterion,
                      args, False)

    # train for one epoch
    train(train_loader, model, ofa_pruner, criterion, soft_criterion, optimizer,
          epoch, args, lr_scheduler)

    # evaluate on validation set
    validate_subnet(train_loader, val_loader, model, ofa_pruner, criterion,
                    args, True)

    if not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
      save_checkpoint({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'best_acc1': best_acc1,
          'optimizer': optimizer.state_dict(),
          'scheduler': lr_scheduler.state_dict(),
      })

def train(train_loader, model, ofa_pruner, criterion, soft_criterion, optimizer,
          epoch, args, lr_scheduler):

  batch_time = AverageMeter('Time', ':6.3f')
  data_time = AverageMeter('Data', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  progress = ProgressMeter(
      len(train_loader), [
          batch_time,
          data_time,
          losses,
          top1,
          top5,
      ],
      prefix="Epoch: [{}]".format(epoch))

  # switch to train mode
  model.train()

  end = time.time()
  for i, (images, target) in enumerate(train_loader):

    # measure data loading time
    data_time.update(time.time() - end)

    if args.gpu is not None:
      images = images.cuda(args.gpu, non_blocking=True)
    if torch.cuda.is_available():
      target = target.cuda(args.gpu, non_blocking=True)

    # total subnets to be sampled
    optimizer.zero_grad()

    args.teacher_model.train()
    with torch.no_grad():
      soft_logits = args.teacher_model(images).detach()

    for arch_id in range(4):
      if arch_id == 0:
        model, _ = ofa_pruner.sample_subnet(model, 'max')
      elif arch_id == 1:
        model, _ = ofa_pruner.sample_subnet(model, 'min')
      else:
        model, _ = ofa_pruner.sample_subnet(model, 'random')

      # calcualting loss
      output = model(images)

      if soft_criterion:
        loss = soft_criterion(output, soft_logits) + criterion(output, target)
      else:
        loss = criterion(output, target)

      loss.backward()

    #clip gradients if specfied
    torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)

    optimizer.step()
    lr_scheduler.step()

    # measure accuracy and record loss
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    losses.update(loss.item(), images.size(0))
    top1.update(acc1[0], images.size(0))
    top5.update(acc5[0], images.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % args.print_freq == 0:
      progress.display(i)
    if i % 1000 == 0:
      print('cur lr: ', lr_scheduler.get_lr()[0])

def validate_subnet(train_loader, val_loader, model, ofa_pruner, criterion,
                    args, bn_calibration):

  subnets_to_be_evaluated = {
      'ofa_min_subnet': {},
      'ofa_max_subnet': {},
      'ofa_random_subnet': {},
  }

  for net_id in subnets_to_be_evaluated:
    if net_id == 'ofa_min_subnet':
      dynamic_subnet, dynamic_subnet_setting = ofa_pruner.sample_subnet(
          model, 'min')
    elif net_id == 'ofa_max_subnet':
      dynamic_subnet, dynamic_subnet_setting = ofa_pruner.sample_subnet(
          model, 'max')
    elif net_id.startswith('ofa_random_subnet'):
      dynamic_subnet, dynamic_subnet_setting = ofa_pruner.sample_subnet(
          model, 'random')
    else:
      dynamic_subnet_setting = subnets_to_be_evaluated[net_id]
      static_subnet, _, flops, params = ofa_pruner.get_static_subnet(
          model, dynamic_subnet_setting)

    if len(subnets_to_be_evaluated[net_id]) == 0:
      static_subnet, _, flops, params = ofa_pruner.get_static_subnet(
          dynamic_subnet, dynamic_subnet_setting)

    static_subnet = static_subnet.cuda()

    if bn_calibration:
      with torch.no_grad():
        static_subnet.eval()
        ofa_pruner.reset_bn_running_stats_for_calibration(static_subnet)

        for batch_idx, (images, _) in enumerate(train_loader):
          if batch_idx >= 16:
            break
          images = images.cuda(non_blocking=True)
          static_subnet(images)  #forward only

    acc1, acc5 = validate(val_loader, static_subnet, criterion, args)

    summary = {
        'net_id': net_id,
        'mode': 'evaluate',
        'acc1': acc1,
        'acc5': acc5,
        'flops': flops,
        'params': params
    }

    print(summary)

def validate(val_loader, model, criterion, args):
  batch_time = AverageMeter('Time', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  progress = ProgressMeter(
      len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

  # switch to evaluate mode
  model.eval()

  with torch.no_grad():
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
      if args.gpu is not None:
        images = images.cuda(args.gpu, non_blocking=True)
      if torch.cuda.is_available():
        target = target.cuda(args.gpu, non_blocking=True)

      # compute output
      output = model(images)
      loss = criterion(output, target)

      # measure accuracy and record loss
      acc1, acc5 = accuracy(output, target, topk=(1, 5))
      losses.update(loss.item(), images.size(0))
      top1.update(acc1[0], images.size(0))
      top5.update(acc5[0], images.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % args.print_freq == 0:
        progress.display(i)

    # TODO: this should also be done with the ProgressMeter
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
        top1=top1, top5=top5))

  return float(top1.avg), float(top5.avg)

"""
It's often necessary to clip the maximum 
gradient value (e.g., 1.0) when using this adaptive KD loss
# Implementation adapted from AlphaNet: https://github.com/facebookresearch/AlphaNet
"""

class AdaptiveLossSoft(torch.nn.modules.loss._Loss):

  def __init__(self, alpha_min=-1.0, alpha_max=1.0, iw_clip=5.0):
    super(AdaptiveLossSoft, self).__init__()
    self.alpha_min = alpha_min
    self.alpha_max = alpha_max
    self.iw_clip = iw_clip

  def f_divergence(self, q_logits, p_logits, alpha, iw_clip=1e3):
    assert isinstance(alpha, float)
    q_prob = torch.nn.functional.softmax(q_logits, dim=1).detach()
    p_prob = torch.nn.functional.softmax(p_logits, dim=1).detach()
    q_log_prob = torch.nn.functional.log_softmax(
        q_logits, dim=1)  #gradient is only backpropagated here

    importance_ratio = p_prob / q_prob
    if abs(alpha) < 1e-3:
      importance_ratio = importance_ratio.clamp(0, iw_clip)
      f = -importance_ratio.log()
      f_base = 0
      rho_f = importance_ratio.log() - 1.0
    elif abs(alpha - 1.0) < 1e-3:
      f = importance_ratio * importance_ratio.log()
      f_base = 0
      rho_f = importance_ratio
    else:
      iw_alpha = torch.pow(importance_ratio, alpha)
      iw_alpha = iw_alpha.clamp(0, iw_clip)
      f = iw_alpha / alpha / (alpha - 1.0)
      f_base = 1.0 / alpha / (alpha - 1.0)
      rho_f = iw_alpha / alpha + f_base

    loss = torch.sum(q_prob * (f - f_base), dim=1)
    grad_loss = -torch.sum(q_prob * rho_f * q_log_prob, dim=1)
    return loss, grad_loss

  def forward(self, output, target, alpha_min=None, alpha_max=None):
    alpha_min = alpha_min or self.alpha_min
    alpha_max = alpha_max or self.alpha_max

    loss_left, grad_loss_left = self.f_divergence(
        output, target, alpha_min, iw_clip=self.iw_clip)
    loss_right, grad_loss_right = self.f_divergence(
        output, target, alpha_max, iw_clip=self.iw_clip)

    ind = torch.gt(loss_left, loss_right).float()
    loss = ind * grad_loss_left + (1.0 - ind) * grad_loss_right

    return loss.mean()

def save_checkpoint(state, filename='checkpoint.pth.tar'):
  torch.save(state, filename)
  shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)

class ProgressMeter(object):

  def __init__(self, num_batches, meters, prefix=""):
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix

  def display(self, batch):
    entries = [self.prefix + self.batch_fmtstr.format(batch)]
    entries += [str(meter) for meter in self.meters]
    print('\t'.join(entries))

  def _get_batch_fmtstr(self, num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
  main()
