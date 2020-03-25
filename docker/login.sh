#!/bin/bash

if [ $USER == "root" ]; then
  echo "Running as ROOT"
  echo ". /opt/vitis_ai/conda/etc/profile.d/conda.sh" >> $HOME/.bashrc
  echo "export PATH=/opt/vitis_ai/conda/bin:$PATH" >> $HOME/.bashrc
  echo "export VERSION=$VERSION" >> $HOME/.bashrc
  echo "export DATE=\"$DATE\"" >> $HOME/.bashrc
  echo "export VAI_ROOT=$VAI_ROOT" >> $HOME/.bashrc
  echo "export PYTHONPATH=$PYTHONPATH" >> $HOME/.bashrc
  echo "/etc/banner.sh" >> $HOME/.bashrc
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
  
  if [ -f /tmp/.Xauthority ]; then
     cp /tmp/.Xauthority /home/$USER/
     chmod -R 600 /home/$USER/.Xauthority
     chown $UID:vitis-ai-users /home/$USER/.Xauthority
  fi
  sudo -H -u $USER $@
  
fi
exit 0
