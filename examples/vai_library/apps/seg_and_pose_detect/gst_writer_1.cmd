appsrc ! videoconvert ! videoscale  !video/x-raw, width=960, height=1080! kmssink driver-name=xlnx plane-id=44 render-rectangle="<960,0,960,1080>" sync=false
