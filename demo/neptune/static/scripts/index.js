/************************
 * Websockets Manager
 ************************/
function WebsocketMgr(cfg) {
  this.url = cfg.url;
  this.callbacks = cfg.callbacks;

  this.ws = null;

  this.init();
}
WebsocketMgr.prototype.init = function() {
  var wsMgr = this;
  this.ws = new WebSocket(this.url);
  var ws = this.ws;

  ws.onopen = function() {
    console.log("[WebsocketMgr] connected to "
      + wsMgr.url);
    function schedule(i) {
      setTimeout(function() {
        //ws.send('Hello from the client! (i=' + i + ')');
        schedule((i + 1) % 1000);
      }, 120000);
    };
    schedule(1);
  };

  ws.onmessage = function(evt)
  {
    var message = evt.data;
    var obj = JSON.parse(message);

    if (wsMgr.callbacks && obj.topic in wsMgr.callbacks)
      wsMgr.callbacks[obj.topic](obj.message)
  };

  ws.onclose = function()
  {
    console.log("[WebsocketMgr] disconnected from "
      + wsMgr.url + ", trying to reconnect in 2 seconds...");
    setTimeout(function() {
      wsMgr.init();
    }, 2000);
  };
}
WebsocketMgr.prototype.send = function(msg) {
  this.ws.send(msg);
}
