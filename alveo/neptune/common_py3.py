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
