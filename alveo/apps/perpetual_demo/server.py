#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import datetime, getopt, json, os, socket, sys, time, zmq
from multiprocessing.pool import Pool
from multiprocessing import Manager, Process

import tornado.httpserver
import tornado.websocket
import tornado.tcpserver
import tornado.ioloop
import tornado.web
from tornado import gen
from tornado.iostream import StreamClosedError

g_httpPort = 8998
g_wsPort = 8999

class MyWebSocketHandler(tornado.websocket.WebSocketHandler):
  clientConnections = []

  def __init__(self, *args, **kwargs):
    super(MyWebSocketHandler, self).__init__(*args, **kwargs)
    print "[WS] websocket ready"

  def open(self):
    print "[WS] websocket opened"
    MyWebSocketHandler.clientConnections.append(self)

  def on_message(self, message):
    print '[WS] message received %s' % message

  def on_close(self):
    print "[WS] websocket closed"
    MyWebSocketHandler.clientConnections.remove(self)

  @staticmethod
  def broadcastMessage(topic, msg):
    try:
      if not msg:
        return

      msgPOD = {}
      msgPOD['time'] = datetime.datetime.now().strftime("%I:%M:%S.%f %p on %B %d, %Y")
      msgPOD['topic'] = topic
      msgPOD['data'] = msg

      print "[WS] broadcast %s to %d client(s)" \
        % (topic, 
           len(MyWebSocketHandler.clientConnections))
      for socket in MyWebSocketHandler.clientConnections:
        socket.write_message(json.dumps(msgPOD))
    except:
      return

  def check_origin(self, origin):
    return True

wsApp = tornado.web.Application([
  (r"/", MyWebSocketHandler),
])


pwd = os.path.dirname(os.path.realpath(__file__))
httpApp = tornado.web.Application([
  (r"/.*val-min/(.*)", tornado.web.StaticFileHandler,
    {'path': "%s/www/imagenet_val" % pwd } ),
  (r"/scratch/(.*)", tornado.web.StaticFileHandler,
    {'path': "/scratch"} ),
  (r"/static/(.*)", tornado.web.StaticFileHandler,
    {'path': "%s" % pwd} )
])

"""
  Subscribe to C++ system for updates. Send updates to GUI websocket
"""
def backgroundZmqCaffeListener(uri, webSocketQ):
  print "Subscribe to C++ updates %s" % uri
  context = zmq.Context()
  socket = context.socket(zmq.SUB)
  socket.connect("tcp://%s" % uri)
  socket.setsockopt(zmq.SUBSCRIBE, "")

  while True:
    try:
      #  Wait for next request from client
      message = socket.recv()
      #print("Received caffe request: %s" % message)
      webSocketQ.put(("caffe", message))
#      MyWebSocketHandler.broadcastMessage("caffe", message)
    except:
      print ("Got an exception in backgroundZmqCaffeListener", uri)
      pass

def backgroundWebSocketQ(webSocketQ):
  while not webSocketQ.empty():
    (topic, msg) = webSocketQ.get()
    MyWebSocketHandler.broadcastMessage(topic, msg)

def backgroundZmqXmlrtListener(uri, webSocketQ):
  print "Subscribe to xMLrt updates %s" % uri
  context = zmq.Context()
  socket = context.socket(zmq.SUB)
  socket.connect("tcp://%s" % uri)
  socket.setsockopt(zmq.SUBSCRIBE, "")

  while True:
    try:
      #  Wait for next request from client
      message = socket.recv()
      #print("Received xmlrt request: %s, %s" % (message, uri))
      webSocketQ.put(("xmlrt", message))
#      MyWebSocketHandler.broadcastMessage("xmlrt", message)
    except:
      print ("Got an exception in backgroundZmqXmlrtListener", uri)
      pass

def listUrls(urlstr, url_list) :
  u_list = urlstr.split(",")
  m_ip = ""
  for i in range(len(u_list)) :
    uri = u_list[i]
    if m_ip == "" :
      m_ip = uri[:uri.find(':')]
    if len(uri) == 4 :
      uri = m_ip + ":" + uri
    if len(uri) > 0 :
      url_list.append(uri)

def main():
  zmqSubUrl = []
  zmqXSubUrl = []

  try:
    opts, args = getopt.getopt(\
      sys.argv[1:], 
      "z:x:", 
      ["zmq_url=", "zmq_xmlrt_url"])
  except getopt.GetoptError as err:
    print(err)
    sys.exit(2)

  for o,a in opts:
    if o == "-z":
      listUrls(a, zmqSubUrl)
    elif o == "-x":
      listUrls(a, zmqXSubUrl)
  
  multiMgr = Manager()
  webSocketQ = multiMgr.Queue(maxsize=1)

  assert len(zmqSubUrl) > 0 and len(zmqXSubUrl) > 0
  assert len(zmqSubUrl) == len(zmqXSubUrl)
  workers = Pool(len(zmqSubUrl) + len(zmqXSubUrl) + 1)
  
  for uri in zmqSubUrl:
    workers.apply_async(backgroundZmqCaffeListener, [uri, webSocketQ])
  for uri in  zmqXSubUrl:
    workers.apply_async(backgroundZmqXmlrtListener, [uri, webSocketQ])

  print "Start websocket server %s:%d" % (socket.gethostname(), g_wsPort)
  wsServer = tornado.httpserver.HTTPServer(wsApp)
  wsServer.listen(g_wsPort)

  print "Start http server on %s:%d" % (socket.gethostname(), g_httpPort)
  httpServer = tornado.httpserver.HTTPServer(httpApp)
  httpServer.listen(g_httpPort)

  tornado.ioloop.PeriodicCallback(\
    lambda: backgroundWebSocketQ(webSocketQ), 500).start()
  tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
  main()
