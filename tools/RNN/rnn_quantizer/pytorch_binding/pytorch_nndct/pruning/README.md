# PyTorch Model Pruning Toolkit
Coarse-grained pruning for PyTorch modules.

## Getting started with Vitis-AI docker
There is a conda environment "vitis-ai-pytorch" in Vitis-AI docker and you can directly start our "resnet18" example without installation steps.
If you want a different python/pytorch/torchvision version, you can also install nndct from source code.
- Copy [resnet18_pruning.py](../../../example/resnet18_pruning.py) example to docker environment
- Download pre-trained [resnet18 model](https://download.pytorch.org/models/resnet18-5c106cde.pth)
  ```shell
  wget https://download.pytorch.org/models/resnet18-5c106cde.pth -O resnet18.pth
  ```
- Prepare Imagenet dataset, see http://image-net.org/download-images
- Run pruning and fine-tuning by specifing the directories where model and images in
  ```shell
  python resnet18_pruning.py --data_dir imagenet_dir --pretrained model_dir/resnet18.pth --ratio 0.1  --ana True
  ```
  
## Installation
For installation instructions, see [this documentation](../../../README.md)

## Model Pruning Workflow
A typical workflow of pruning includes:
1. Preparing a baseline model
2. Creating a pruner
3. Performing model analysis
4. Pruning the model
5. Fine-tuing the pruned model
6. Back to step 4

### Prepare a baseline model
We use the resnet18 pretrained model from torchvision.
```python
from torchvision.models.resnet import resnet18

model = resnet18()
model.load_state_dict(torch.load('resnet18.pth'))
```

### Create a pruner
A pruner can be created by providing the model to be pruned and its input shape and input dtype.

**Note:** The shape is the size of the input image and does not contain batch size.
```python
from pytorch_nndct import Pruner
from pytorch_nndct import InputSpec

pruner = Pruner(model, InputSpec(shape=(3, 224, 224), dtype=torch.float32))
```
For models with multiple inputs, you can use a list of InputSpec to initialize a pruner.

### Run model analysis
To run model analysis, you need to define a function that can be used to evaluate the model.
The first argument of this function must be the model to be evaluated.
```python
def evaluate(val_loader, model, criterion):
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
      model = model.cuda()
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)

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

      if i % 50 == 0:
        progress.display(i)

    # TODO: this should also be done with the ProgressMeter
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
        top1=top1, top5=top5))

  return top1.avg, top5.avg

def ana_eval_fn(model, val_loader, loss_fn):
  return evaluate(val_loader, model, loss_fn)[1]
```

Then we call ana() method with the function defined above as the first argument.
```
pruner.ana(ana_eval_fn, args=(val_loader, criterion), gpus=[0, 1, 2, 3])
```
The ‘args’ is the tuple of arguments starting from the second argument required by ‘ana_eval_fn’.

### Prune the model
Prune the model by giving a pruning ratio, that means, what percentage of FLOPs is expected to be reduced.
```python
model = pruner.prune(ratio=0.1)
```

### Fine-tune the pruned model
```python
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)

best_acc5 = 0
epochs = 10
for epoch in range(epochs):
  train(train_loader, model, criterion, optimizer, epoch)

  acc1, acc5 = evaluate(val_loader, model, criterion)

  # remember best acc@1 and save checkpoint
  is_best = acc5 > best_acc5
  best_acc5 = max(acc5, best_acc5)

  if is_best:
    model.save('resnet18_sparse.pth.tar')
    torch.save(model.state_dict(), 'resnet18_final.pth.tar')

```
Note that in the last two lines of code we save two checkpoint files.
‘model.save()’ saves sparse weights with the same shapes as the baseline model and the removed channel is set to 0.
‘model.state_dict()’ returns dense weights with the pruned shapes.
The first checkpoint is used as input for the next round of pruning, and the second checkpoint is used for the final deployment.
That is, if there is another round of pruning, then use the first checkpoint, and if this is the last round of pruning, then use the second checkpoint.

### Iterative pruning
If too many parameters are removed from the model at once, it will be difficult to recover the model accuracy. 
The reduction FLOPs is gradually increased in every iteration, to help better recover accuracy during the finetune stage.
Pruning followed by retraining forms one iteration. 
In the first iteration of pruning, the input model is the baseline model, and it will be pruned and fine-tuned.
In subsequent iterations, the fine-tuned model obtained from previous iteration is used to prune again.
Such process is usually repeated several times until a desired sparse model is obtained.

Load the sparse checkpoint and increase pruning ratio. Here we increase the pruning ratio from 0.1 to 0.2.
```python
model = resnet18()
model.load_state_dict(torch.load('resnet18_best.pth.tar'))

pruner = Pruner(model, InputSpec(shape=(3, 224, 224), dtype=torch.float32))
model = pruner.prune(ratio=0.2)
```
When we get the new pruned model, we can start fine-tuning again.

## Pruning APIs
### pytorch_nndct.InputSpec
#### Attributes
| Name | Description |
| ------ | ------ |
| shape | Shape tuple, expected shape of the input. |
| dtype | Expected [torch.dtype](https://pytorch.org/docs/stable/tensor_attributes.html?highlight=dtype#torch.torch.dtype) of the input. |

### pytorch_nndct.Pruner
| Arguments | Description |
| ------ | ------ |
| module | A torch.nn.Module object to be pruned. |
| input_specs | The inputs of the module: a InputSpec object or list of InputSpec |

#### Methods
##### ana
| Arguments | Description |
| ------ | ------ |
| eval_fn | Callable object that takes a [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) object as its first argument and returns the evaluation score. |
| args | A tuple of arguments that will be passed to eval_fn. |
| gpus | A tuple or list of GPU indices used for model analysis. If not set, the default GPU will be used. |

##### prune
| Arguments | Description |
| ------ | ------ |
| ratio | The expected percentage of FLOPs reduction. This is just a hint, the actual FLOPs drop not necessarily strictly to this value after pruning. |
| threshold | Relative proportion of model performance loss that can be tolerated. |
| excludes | Modules that need to prevent from pruning. |
| output_script | Filepath that saves the generated script used for rebuilding model. |

##### summary
| Arguments | Description |
| ------ | ------ |
| pruned_model | A pruned module returned by [prune()](#prune) method. |
