appsrc ! videoconvert ! videoscale  !video/x-raw, width=960, height=1080! kmssink driver-name=xlnx plane-id=42 render-rectangle="<0,0,960,1080>" sync=false
