appsrc ! videoconvert ! videoscale  !video/x-raw, width=640, height=480 ! kmssink driver-name=xlnx plane-id=34 render-rectangle="<1200,252,640,480>" sync=false
