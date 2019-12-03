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
import asyncio
import tornado

# use this to launch: https://stackoverflow.com/a/46710785
# this code from here: https://gist.github.com/nvgoldin/30cea3c04ee0796ebd0489aa62bcf00a
# the async def syntax raises an error in Py2. For Py2/Py3 compatibility, the
# the main script imports the appropriate one
def cancel_async_tasks():
    async def shutdown(loop):
        tasks = [task for task in asyncio.Task.all_tasks() if task is not
                asyncio.tasks.Task.current_task()]
        list(map(lambda task: task.cancel(), tasks))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()

    loop = tornado.ioloop.IOLoop.instance()
    asyncio.ensure_future(shutdown(loop))
