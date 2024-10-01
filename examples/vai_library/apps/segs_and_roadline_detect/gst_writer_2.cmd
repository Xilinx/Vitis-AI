appsrc ! videoconvert ! videoscale  !video/x-raw, width=512, height=288 ! kmssink driver-name=xlnx plane-id=46 render-rectangle="<100,540,512,288>" sync=false
