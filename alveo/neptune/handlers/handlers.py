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
import json
import logging
import os
import random
import time
import tornado.ioloop

import numpy as np

from neptune.service_manager import ServiceManager

logger = logging.getLogger(__name__)

async def image_handler(self):
    requestId = self.application.request_id_gen.get()
    img = self.get_argument('image', None)
    img_format = self.get_argument('dtype', 'uint8')
    url = self.get_argument('url', None)
    meta = {
        'id': requestId,
        'status': -1
    }

    if img is not None:
        dat = json.loads(img)
        image = np.asarray(dat["data"], dtype=getattr(np, img_format))
        meta['dtype'] = img_format
        meta['mode'] = "image"
    elif url is not None:
        meta['url'] = url
        meta['mode'] = 'url'
        meta['dtype'] = img_format
        image = np.zeros(0)
    else:
        self.write(json.dumps(meta))
        return

    loop = tornado.ioloop.IOLoop.instance()
    result = await loop.run_in_executor(ServiceManager()._threadpool,
        ServiceManager().run, self.handlers['service'], image, meta)
    if result is not None:
        (meta, buf) = result

    # print(json.dumps(meta))
    self.write(json.dumps(meta))

async def video_handler(self):
    requestId = self.application.request_id_gen.get()
    img_format = self.get_argument('dtype', 'uint8')
    url = self.get_argument('url', None)
    mode = self.get_argument('mode', None)
    callback_id = self.get_argument('callback_id', None)
    video_type = self.get_argument('video_type', 'live')

    meta = {
        'id': requestId,
        'status': -1
    }

    if url is not None:
        meta['url'] = url
        meta['dtype'] = img_format
        meta['video_type'] = video_type
        if callback_id:
            meta['callback_id'] = callback_id
    else:
        self.write(json.dumps(meta))
        return

    # print("%s:%f Starting service" % (callback_id, time.time()))
    loop = tornado.ioloop.IOLoop.instance()
    result = await loop.run_in_executor(ServiceManager()._threadpool,
        ServiceManager().run, self.handlers['service'], np.zeros(0), meta)
    # print("%s:%f:Returned from service" % (callback_id, time.time()))
    if result is not None:
        (meta, buf) = result
    else:
        # print("No return from video_handler")
        pass

    # print(json.dumps(meta))
    self.write(json.dumps(meta))

async def ping_handler(self):
    meta = { 'id': self.application.request_id_gen.get() }
    #callback_id = self.get_argument('callback_id', None)
    #if callback_id:
    #    meta['callback_id'] = callback_id

    loop = tornado.ioloop.IOLoop.instance()
    result = await loop.run_in_executor(ServiceManager()._threadpool,
        ServiceManager().run, self.handlers['service'], np.zeros(1), meta)
    # print("Returned from service_manager run")
    if result:
        (meta, buf) = result
        self.write(json.dumps(meta, sort_keys=True, indent=4,
            separators=(',',':'), ensure_ascii=False))
    else:
        logger.warning("Nothing returned from ping handler")
        self.set_status(500)
        self.write("Ping service is not active")
