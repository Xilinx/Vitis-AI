#!/usr/bin/python3

# Copyright 2019 Xilinx Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
import flask
import random
import time
import threading
import json
import argparse
import fcntl
from subprocess import Popen, PIPE
from multiprocessing import Process


###################BE Manager######################
class BETask:
    def __init__(self, _id, _dataPath, _filename=""):
        self.id = _id
        self.data = _dataPath
        self.filename = _filename
        """BETasl.status can be ['inited', 'started', 'finished', 'error']"""
        self.status = 'inited'
        self.info = ""
        self.taskHandle = None

    def __str__(self):
        return "[Task Id]: %s, [Data File]: %s, [Status]: %s\n" %\
            (self.id, self.data, self.status)

    def check(self):
        pass


class BEManager:
    def __init__(self, BEPath: str, dbPath: str = ""):
        self.BEExePath = os.path.abspath(BEPath)
        self.BEWaitQ = []
        self.BERunQ = []
        self.running = False
        self.proc = None
        self.idFnList = []

        try:
            with open(dbPath, 'r') as fpDB:
                json.loads(fpDB.read())
        except BaseException:
            os.system('mkdir -p ./data')
            with open(dbPath, 'w+') as fpDB:
                print("Re-init fildname db !!!")
                fpDB.write("{}")
        self.database = dbPath

    def manageDB(self, ops: str, id=0, filename=""):
        with open(self.database, 'r+') as fpDB:
            idFnMap = json.loads(fpDB.read())
            ops == ops.lower()

            fcntl.flock(fpDB, fcntl.LOCK_EX)

            if ops == "add":
                idFnMap[id] = filename
            elif ops == "del" or ops == "rm":
                if id in idFnMap.keys():
                    idFnMap.pop(id)
                    os.system('rm ./data/%s ./data/%s.json' % (id, id))
            elif ops == "update":
                existIDs = []
                for r, d, files in os.walk('./data'):
                    for f in files:
                        if f.endswith('.json'):
                            id = f.strip('.json')
                            rawFn = idFnMap.get(id)
                            ctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getctime(os.path.join(r, f))))
                            existIDs.append({'ctime': ctime, 'id': id, 'fn': rawFn})

                existIDs.sort(key=lambda x: x['ctime'], reverse=True)
                self.idFnList = existIDs

            fpDB.seek(0)
            fpDB.write(json.dumps(idFnMap, indent=1))
            fpDB.truncate()
            fcntl.flock(fpDB, fcntl.LOCK_UN)

    def checkProfilerBE(self, path):
        return os.path.exists(path)

    def runTask(self, task: BETask):
        pass

    def runBE(self):
        while(self.running):
            if len(self.BEWaitQ) == 0:
                time.sleep(0.1)
            else:
                """Get task from WaitQ"""
                task = self.BEWaitQ.pop(0)
                self.BERunQ.append(task)
                print(task)

                """Prepare to start task"""
                id = task.id
                args = [task.data, task.data + '.json']
                exec = self.BEExePath

                """Start Subprocess"""
                task.taskHandle = Popen(['python3', exec, *args], stdout=PIPE, stderr=PIPE)
                task.status = 'started'

    def run(self):
        self.running = True
        p = threading.Thread(target=self.runBE)
        self.proc = p
        p.start()

    def stop(self):
        self.running = False
        self.proc.join()

    def addBETask(self, id: str, dataPath, filename=""):
        self.manageDB('add', id, filename)
        self.BEWaitQ.append(BETask(id, dataPath, filename))

    def getBETask(self, id: str) -> BETask:
        pass

    def delBETask(self, id):
        pass

    """
    Return NO: task not found,
           OK: task success
           WAIT: task on wait queue
           RUN: task running
    """

    def checkBETask(self, id: str) -> dict:
        print("Check ID %s" % id)
        taskStatus = {'status': 'notask', 'info': None, 'retcode': None}

        for t in self.BERunQ:
            if t.id == id:
                taskStatus['status'] = t.status
                info = t.taskHandle.stdout.readline()
                info = info.decode()
                taskStatus['info'] = [info]
                rc = t.taskHandle.poll()
                if rc is None:
                    taskStatus['retcode'] = -1
                else:
                    """Task finished, read out all the rest log"""
                    restLines = t.taskHandle.stdout.readlines()
                    restLines = [i.decode() for i in restLines]
                    taskStatus['info'] += restLines
                    taskStatus['retcode'] = rc
                    taskStatus['status'] = 'finished'
            else:
                continue

        return taskStatus


###################WebAPP#######################

app = flask.Flask(__name__)
profilerBE = BEManager('./xatAnalyzer/analyzer.py', './data/db')


@app.route('/process', methods=['GET', 'POST'])
def doProcess():
    taskId = flask.request.args.get('id')

    if flask.request.method == 'GET':

        print("Get Request of Process, id=%s" % taskId)

        return flask.render_template('process.html')

    if flask.request.method == 'POST':
        respon = profilerBE.checkBETask(taskId)
        return json.dumps(respon)


@app.route('/manage', methods=['GET'])
def manage():
    # if flask.request.method == 'GET':
    op = flask.request.args.get('op')
    id = flask.request.args.get('id')

    if op == 'del':
        profilerBE.manageDB('del', id)
        return flask.redirect(flask.url_for('manage'))

    profilerBE.manageDB('update')
    l = profilerBE.idFnList
    return flask.render_template('manage.html', flist=l)


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if flask.request.method == 'GET':
        demoList = [f.split('.')[0] for f in list(os.walk('static/examples/'))[0][2] if f.endswith('xat')]
        debList = [f.split('.')[0] for f in list(os.walk('static/examples/'))[0][2] if f.endswith('deb')]
        return flask.render_template('upload.html', hostip=HOSTIP, port=PORT, flist=demoList, downloadDeb=(len(debList) != 0))

    if flask.request.method == 'POST':
        f = flask.request.files['file']

        while True:
            randId = str(random.randint(100000, 999999))
            path = os.path.join('./data', randId)
            """If taskId existed, re-generate one"""
            if not os.path.exists(path):
                break

        f.save(path)
        print("Saved %s, Id %s" % (f.filename, randId))
        profilerBE.addBETask(randId, os.path.abspath(path), f.filename)

        return flask.redirect(flask.url_for('doProcess', id=randId))


@app.route('/show')
def doShow():
    taskId = flask.request.args.get('id')

    if flask.request.method == 'GET':
        return flask.render_template('profiler.html')


@app.route('/doc')
def showDoc():
    docId = flask.request.args.get('id')

    try:
        mdText = open('./docs/%s.md' % docId.strip(), 'r').read()
    except BaseException:
        mdText = "Error: Document not available."

    if (flask.request.args.get('get')):
        return mdText

    return flask.render_template('md_doc.html')


@app.route('/result/<taskId>')
def getResult(taskId):
    try:
        with open('./data/%s.json' % taskId) as r:
            return r.read()
    except BaseException:
        return "Error: No such file"


@app.route('/rawfile')
def getRawFile():
    taskId = flask.request.args.get('id')
    f = "./data/%s" % taskId
    fp = open(f, 'rb')
    print("filename %s" % f)
    if not os.path.isfile(f):
        return flask.redirect('/manage')

    profilerBE.manageDB('update')
    fList = profilerBE.idFnList
    fName = "%s.xat" % taskId
    """Looking for the proper file name"""
    for f in fList:
        if f['id'] == taskId:
            fName = f['fn']

    return flask.make_response(
        flask.send_file(
            fp,
            as_attachment=True,
            cache_timeout=0,
            attachment_filename=fName,
            mimetype="application/x-compress"))


@app.route('/')
def index():
    return flask.redirect(flask.url_for('upload'))


@app.route('/examples/<name>')
def getExampleFile(name):
    if name == 'pkg':
        #latestPKG = os.popen("cd vaitrace;make update").read().strip()
        latestPKG = "./static/examples/vaitrace.deb"
        return flask.make_response(flask.send_file(latestPKG, as_attachment=True, mimetype="application/x-compress"))

    f = "./static/examples/%s.xat" % name
    return flask.make_response(flask.send_file(f, as_attachment=True, mimetype="application/x-compress"))


if __name__ == '__main__':
    profilerBE.run()

    parser = argparse.ArgumentParser(prog="Xilinx Vitis AI Profiler")
    parser.add_argument("-i", "--ip", dest="ip", nargs="?", default="0.0.0.0",
                        help="Xilinx Vitis AI Profiler Server IP address, default: [localhost]")
    parser.add_argument("-p", "--port", dest="port", nargs="?", default=8008, help="Xilinx Vitis AI Profiler Server Port, default: [8008]")
    args = parser.parse_args()

    HOSTIP = args.ip
    PORT = args.port
    print(HOSTIP)

    if not os.path.exists('./data'):
        os.system("mkdir data")
    app.run(HOSTIP, port=int(PORT), debug=True)
