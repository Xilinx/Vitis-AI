appsrc ! videoconvert ! videoscale  !video/x-raw, width=512, height=288 ! kmssink driver-name=xlnx plane-id=42 render-rectangle="<100,252,512,288>" sync=false
