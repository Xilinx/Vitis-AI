#!/usr/bin/python
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

# logging should be setup first so imported modules' logging is configured too
import os
from vai.dpuv1.rt import logging_mp
log_file = os.environ['VAI_ALVEO_ROOT'] + "/neptune/logging.ini"
logging_mp.setup_logger(log_file, 'neptune')

from datetime import datetime
import json
import signal
import threading
import time
import tornado.ioloop
import tornado.web
import tornado.websocket
import uuid
import importlib
import logging
import six
if six.PY3:
    import asyncio

import numpy as np
from tornado.options import define, options, parse_command_line
if six.PY3:
    from tornado.platform.asyncio import AnyThreadEventLoopPolicy

from neptune.common import DummyClass, list_submodules, hinted_tuple_hook
if six.PY3:
    from neptune.common_py3 import cancel_async_tasks
else:
    from neptune.common import cancel_async_tasks
import neptune.construct as construct
from neptune.service_manager import ServiceManager
from neptune.node_manager import NodeManager
from vai.dpuv1.rt import xstream

define("port", default=8998, help="run web server on this port", type=int)
define("wsport", default=8999, help="run websocket server on this port", type=int)
define("debug", default=True, help="run in debug mode")

logger = logging.getLogger(__name__)

class IndexHandler(tornado.web.RequestHandler):
    def get(self, *args):
        self.render("index.html", wsport=options.wsport)

class ListServiceHandler(tornado.web.RequestHandler):
    def get(self, *args):
        ret = {'services': []}
        try:
            services_list = ServiceManager().list()
            services = []
            for name, svc in services_list.items():
                services.append({
                    'name': name,
                    'state': svc['state'],
                    'url': svc['url'],
                    'throughput': svc['throughput']
                })
            services.sort(key=lambda x: x['name'])
            ret = {'services': services}
        except Exception as e:
            logger.exception("List service error")
        self.write(json.dumps(ret, sort_keys=True))

class QueryServiceHandler(tornado.web.RequestHandler):
    def get(self, *args):
        service_name = self.get_argument("service")
        ret = {}
        try:
            services_list = ServiceManager().list()
            if service_name in services_list:
                ret = services_list[service_name]
                del ret['service']
        except Exception as e:
            logger.exception("Query service error")
        self.write(json.dumps(ret, sort_keys=True))

class StartServiceHandler(tornado.web.RequestHandler):
    def get(self, *args):
        service_name = self.get_argument("id")
        runtime_args = self.get_argument("args", {})

        ServiceManager().start(service_name, runtime_args)
        self.write("service started")

    def post(self, *args):
        data = json.loads(self.request.body)

        service_name = data["id"] # in unicode
        runtime_args = data["args"] if 'args' in data else {}

        ServiceManager().start(service_name, runtime_args)

        self.write("service started")

class StopServiceHandler(tornado.web.RequestHandler):
    def get(self, *args):
        service_name = self.get_argument("id")
        ServiceManager().stop(service_name)
        self.write("service stopped")

class ConstructServiceHandler(tornado.web.RequestHandler):

    def get(self, *args):
        self.write("POST a recipe to this address to construct it")

    def post(self, *args):
        """
        Parse JSON arguments in POST body. In Requests module, use json key to
        pass in arguments, not data (which may clobber JSON objects)
        """
        recipe = json.loads(self.request.body, object_hook=hinted_tuple_hook)

        tornado_handler = construct.construct(recipe)
        name = str(recipe['name'])
        url = str(recipe['url'])

        self.application.add_handlers(
            r".*",  # match any host
            tornado_handler
        )
        self.write("service %s constructed at /serve/%s" % (name, url))

# FIXME clients being able to destroy services others may be using is problematic
class DestructServiceHandler(tornado.web.RequestHandler):

    def _destroy(self):
        url = str(self.get_argument("url"))
        name = str(self.get_argument("name"))

        fake_request = DummyClass()
        found_index = -1

        # There's no method in Tornado to delete an added handler currently.
        # By examining the source, the code below works to do so. If Tornado
        # adds an API to do this, this code should be replaced. It may also
        # break with changes to Tornado (tested with Tornado 5.1.1 on 8/15/2019)
        for index, parent_rule in enumerate(self.application.default_router.rules):
            rules = parent_rule.target.rules
            for rule in rules:
                fake_request.path = r"/serve/" + url
                if isinstance(rule.matcher.match(fake_request), dict):
                    found_index = index

        return found_index, name, url

    def get(self, *args):
        self.write("POST the name and url of the service to destroy") # TODO only should specify one
    def post(self, *args):
        found_index, name, url = self._destroy()

        if found_index != -1:
            del self.application.default_router.rules[found_index]
            ServiceManager().remove(name)
            recipe_cache = os.environ["VAI_ALVEO_ROOT"] + "/neptune/recipes/recipe_%s.bak" % name
            os.remove(recipe_cache)
            self.write("service destroyed at /serve/%s" % url)
        else:
            self.write("Service %s cannot be destroyed as it does not exist" % name)

class RenderHandler(tornado.web.RequestHandler):

    def get(self, *args):
        url = self.request.uri
        url_arg = url.split('/')[-1]
        html = url_arg + ".html"
        self.render(html, wsport=options.wsport)

class RequestIdGenerator(object):
    def __init__(self):
        self.handler_ids = {}

    def get(self, name='__default__'):
        if name not in self.handler_ids:
            self.handler_ids[name] = 0

        curr_id = self.handler_ids[name]
        self.handler_ids[name] = (curr_id + 1) % 10000 # wraparound
        return curr_id

class WebSocketHandler(tornado.websocket.WebSocketHandler):
    clientConnections = []

    def __init__(self, *args, **kwargs):
        super(WebSocketHandler, self).__init__(*args, **kwargs)
        print("[WS] websocket ready")

    def open(self):
        self.id = str(uuid.uuid4())
        self.last_send = None

        print("[WS] websocket opened %s" % self.id)
        self.send('id', self.id)
        WebSocketHandler.clientConnections.append(self)

    def on_message(self, messageStr):
        try:
            print('[WS] message received from %s: %s' % (self.id, messageStr))
            message = json.loads(messageStr)
            if message['topic'] == 'update_id':
                origId = message['id']
                self.id = origId # take over original id
        except:
            pass

    def on_close(self):
        print("[WS] websocket closed %s" % self.id)
        WebSocketHandler.clientConnections.remove(self)

    def send(self, topic, msg):
        if not msg:
            return

        now = time.time()
        if self.last_send and (now - self.last_send) < 0.05:
            # don't flood the client with too many messages; drop
            return
        self.last_send = now

        try:
            msg_POD = {}
            msg_POD['time'] = datetime.now().isoformat()
            msg_POD['topic'] = topic
            msg_POD['message'] = msg
            self.write_message(json.dumps(msg_POD))
        except Exception as e:
            print(e)

    @staticmethod
    def send_to_client(id, topic, msg):
        try:
            for c in WebSocketHandler.clientConnections:
                if c.id == id:
                    c.send(topic, msg)
        except:
            pass

    @staticmethod
    def broadcast(topic, msg):
        try:
            for c in WebSocketHandler.clientConnections:
                c.send(topic, msg)
        except:
            pass

    def check_origin(self, origin):
        return True

class ServerWebApplication(tornado.web.Application):
    def __init__(self):
        self.request_id_gen = RequestIdGenerator()
        handlers = self.init_handlers()

        super(ServerWebApplication, self).__init__(
            handlers,
            cookie_secret="COOKIE_SECRET",
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            xsrf_cookies=False, #? should be true?
            autoreload=False,
            debug=options.debug
        )

    def init_handlers(self):
        """
        Define the basic REST handlers. These cannot be destroyed.

        Returns:
            List: List of handler tuples for initializing a Tornado web app
        """
        handlers = []
        handlers.append((r"/", IndexHandler))

        handlers.append((r"/services/list", ListServiceHandler))
        handlers.append((r"/services/query", QueryServiceHandler))
        handlers.append((r"/services/start", StartServiceHandler))
        handlers.append((r"/services/stop", StopServiceHandler))
        handlers.append((r"/services/construct", ConstructServiceHandler))
        handlers.append((r"/services/destruct", DestructServiceHandler))
        handlers.append((r"/render/([^/]+)", RenderHandler))
        return handlers

class ServerApp(object):
    def __init__(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        # signal.signal(signal.SIGQUIT, self.sigquit_handler)
        self.do_exit = False
        self.hbeat_id = 0
        self.hbeat = 0
        # self.do_restart = False

        self.xserver = xstream.Server()
        parse_command_line()

        self.web_app = ServerWebApplication()
        # Add handlers for default services here so they can be destroyed if
        # needed. Handlers installed in the web_app __init__ cannot be destroyed.
        recipes = self.get_recipes()
        for recipe in recipes:
            tornado_handler = construct.construct(recipe)

            self.web_app.add_handlers(
                r".*",  # match any host
                tornado_handler
            )
        self.web_server = self.web_app.listen(options.port)
        self.ws_app = tornado.web.Application([(r"/", WebSocketHandler)])
        self.ws_server = self.ws_app.listen(options.wsport)

        self.xspub = xstream.Publisher()
        self.xs2server = threading.Thread(target=ServerApp.xstream2server)
        self.xs2server.start()
        self.heartbeat_thread = threading.Thread(target=ServerApp.heartbeat, args=(lambda: self.do_exit,))
        self.heartbeat_thread.start()

    @staticmethod
    def get_recipes():
        """
        Get all recipes from the recipes folder. If recipe_*.bak files exist,
        use those to construct services. Otherwise, construct from source using
        the default recipe functions.
        """
        recipes = []
        recipes_cache = os.environ["VAI_ALVEO_ROOT"] + "/neptune/recipes/"
        file_names = [fn for fn in os.listdir(recipes_cache)
            if fn.startswith('recipe_') and fn.endswith('.bak')]
        if file_names:
            logger.info("Constructing services from cache")
            for file_name in file_names:
                with open(recipes_cache + file_name) as f:
                    recipes.append(json.load(f, object_hook=hinted_tuple_hook))
        else:
            logger.info("Constructing services from source")
            modules = list_submodules('neptune.recipes')
            for module_path in modules:
                module = importlib.import_module(module_path)
                attrs = dir(module)
                for attr in attrs:
                    if attr.startswith('recipe_'):
                        recipe = getattr(module, attr)
                        if callable(recipe):
                            recipes.append(recipe().to_dict())
        return recipes

    @staticmethod
    def xstream2server():
        xs = xstream.Subscribe("__server__")
        while True:
            # subscribe to special "__server__" channel for
            # other processes to send messages to this server
            # e.g. speedodata -> websockets
            msg_str = xs.get_msg()
            if msg_str is None:
                break

            try:
                msg = json.loads(msg_str)
                if msg['topic'] == 'speedodata':
                    WebSocketHandler.broadcast(msg['topic'], msg['message'])
                elif msg['topic'] == 'callback' and 'callback_id' in msg:
                    # print("sending callback message")
                    # print(msg['message'])
                    WebSocketHandler.send_to_client(\
                        msg['callback_id'], msg['topic'], msg['message'])
                elif msg['topic'] == 'xs_throughput':
                    report = json.loads(msg['message'])
                    #print(report)
                    for name, throughput in report.items():
                        serviceName = name.split('.')[0]
                        edgeName = name[name.find('.')+1:]
                        ServiceManager().update_throughput_stats(serviceName,
                            edgeName, throughput)
            except:
                pass

        cancel_async_tasks()

    @staticmethod
    def heartbeat(stop):
        xs = xstream.Subscribe("__heartbeat__", timeout=5000)
        service_manager = ServiceManager()
        node_status = {}

        def check_services(node_status):
            if stop:
                return
            invalid_services = []
            for service, status in node_status.items():
                last_valid = status['last_valid']
                service_state = service_manager._services[service]['state']
                is_starting = service_state == service_manager.STARTING
                is_started = service_state == service_manager.STARTED

                # if the service has been stopped, clear it
                if service_state == service_manager.STOPPED:
                    invalid_services.append(service)
                # if there's a discrepancy in what the service_manager says
                # and what we have cached, clear it
                elif is_starting and node_status[service]['is_started']:
                    invalid_services.append(service)
                # if it's started and hasn't been valid in the last n secs,
                # restart it
                elif is_started and now - last_valid > 5:
                    logger.warning("Service %s is dead, restarting" % service)
                    service_manager.stop(service)
                    service_manager.start(service)
                    node_status[service]['is_started'] = False

            for service in invalid_services:
                del node_status[service]

        logger = logging.getLogger(__name__)

        while True:
            if stop():
                break
            # when enabling coverage, this line will raise an exception for some
            # reason. For now, just catching it
            try:
                msg_str = xs.get_msg()
                now = time.time()
            except Exception:
                logger.exception("Shouldn't happen")

            # the get_msg timed out, i.e. no heartbeats received
            if msg_str == (None, None):
                check_services(node_status)
                continue
            msg = json.loads(msg_str)
            service = msg['service']
            channel = msg['channel']

            # if this is the first time we've seen this service
            if service not in node_status:
                _first_edge, last_edge = service_manager._get_graph_io(service)
                node_status[service] = {
                    'last_valid': 0, # saves the last time this service was valid
                    'is_started': False, # our check that services haven't stopped
                    'last_edge': last_edge[0], # saves the last edge of the service
                    'channels': {} # save heartbeat times for each channel
                }
            node_status[service]['channels'][channel] = now
            service_state = service_manager._services[service]['state']
            if node_status[service]['last_edge'] == channel:
                if service_state == service_manager.STARTING:
                    if not node_status[service]['is_started']:
                        service_manager._services[service]['state'] = service_manager.STARTED
                        node_status[service]['is_started'] = True
                    else:
                        # there's a discrepancy. For example, the service may
                        # have been stopped and something else started with
                        # the same name. In this case, clear the cache
                        del node_status[service]
                        continue
                node_status[service]['last_valid'] = now

            check_services(node_status)

        cancel_async_tasks()

    def launch(self):
        tornado.ioloop.PeriodicCallback(self.check_exit, 500).start()
        loop = tornado.ioloop.IOLoop.instance()
        loop.start()
        loop.close()

    def signal_handler(self, signum, frame):
        logger.status("Shutting down server...")
        ServiceManager().stop_all()
        self.do_exit = True

    # def sigquit_handler(self, signum, frame):
    #     print("restarting server...")
    #     ServiceManager().stop_all()
    #     self.do_restart = True

    def check_exit(self):
        if self.do_exit:
            self.xspub.end("__server__")
            self.xs2server.join()
            self.heartbeat_thread.join()
            self.ws_server.stop()
            self.web_server.stop()
            cancel_async_tasks()
            del self.xserver
        elif self.hbeat > 4:
            self.hbeat = 0
            service_manager = ServiceManager()
            services_list = service_manager.list()
            started_services = []
            for name, svc in services_list.items():
                if svc['state'] >= service_manager.STARTING:
                    started_services.append(name)
            for service in started_services:
                msg = json.dumps({
                    '__heartbeat__': service,
                    'id': 'hbeat_' + str(self.hbeat_id)
                })
                service_manager.send(service, 0, np.zeros(1), msg)
                self.hbeat_id = (self.hbeat_id + 1) % 9999
        else:
            self.hbeat += 1
        # the restarted server is unresponsive. We don't need this functionality
        # right now but if needed, it needs to be debugged.
        # if self.do_restart:
        #     self.xspub.end("__server__")
        #     self.xs2server.join()
        #     self.ws_server.stop()
        #     self.web_server.stop()
        #     del self.xserver
        #     tornado.ioloop.IOLoop.instance().stop()
        #     self.do_restart = False

        #     ServiceManager()._drop()
        #     NodeManager()._drop()
        #     xstream.Statistics()._drop()

        #     main()

def main():
    logging_directory = os.environ['VAI_ALVEO_ROOT'] + '/neptune/logs/'
    log_name = '_{:%Y-%m-%d}.log'.format(datetime.now())
    with open(logging_directory + 'neptune' + log_name, 'a') as f:
        f.write("################################################################\n")

    # https://github.com/tornadoweb/tornado/issues/2531
    if six.PY3:
        asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())

    app = ServerApp()
    app.launch()

if __name__ == "__main__":
    main()
