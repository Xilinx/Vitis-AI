
# Build

```sh
cd vart
./docker_run.sh <docker-image>

cd /workspace
./cmake.sh --cmake-options="-DENABLE_RNN_RUNNER=ON"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/Ubuntu.18.04.x86_64.Debug/lib

# U50
sudo cp /workspace/rnn-runner/xclbin/u50lv/dpu.xclbin /usr/lib/dpu.xclbin

# U25
sudo cp /workspace/rnn-runner/xclbin/u25/dpuv2.xclbin /usr/lib/dpu.xclbin
```
