#!/bin/bash

# Copyright 2019 Xilinx Inc.
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

if [ -z $USER ]; then
  echo "Running as ROOT"
  exec $@
else
  echo "Running as $USER"
  groupadd vitis-ai-users -g $GID -f && \
  useradd -g vitis-ai-users -ms /bin/bash $USER -u $UID && \
  usermod -aG sudo $USER && \
  passwd -d $USER
  echo ". /opt/vitis_ai/conda/etc/profile.d/conda.sh" >> /home/$USER/.bashrc
  echo "export PATH=/opt/vitis_ai/conda/bin:$PATH" >> /home/$USER/.bashrc
  echo "export VERSION=$VERSION" >> /home/$USER/.bashrc
  echo "export DATE=\"$DATE\"" >> /home/$USER/.bashrc
  echo "export VAI_ROOT=$VAI_ROOT" >> /home/$USER/.bashrc
  echo "export PYTHONPATH=$PYTHONPATH" >> /home/$USER/.bashrc
  echo "/etc/banner.sh" >> /home/$USER/.bashrc
  sudo -H -u $USER $@
fi
exit 0
