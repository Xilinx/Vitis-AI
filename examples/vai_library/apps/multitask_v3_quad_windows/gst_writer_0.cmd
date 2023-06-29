appsrc ! videoconvert ! videoscale  !video/x-raw, width=1920, height=1080! kmssink driver-name=xlnx plane-id=42 render-rectangle="<0,0,1920,1080>" sync=false
