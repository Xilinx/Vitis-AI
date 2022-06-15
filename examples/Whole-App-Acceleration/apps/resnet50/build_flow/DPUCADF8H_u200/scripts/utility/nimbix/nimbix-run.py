#!/usr/bin/env python
import sys
import optparse

import os
import pwd
import datetime
import time

import subprocess
import json

import urllib2
import urllib
import re

HEADER = {"Content-Type": "application/json"}
API_URL = "https://api.jarvice.com/jarvice"

def set_job_name_or_number(params, job_name, job_number):
	if job_number is not None:
		params['number'] = job_number
		if job_name is not None:
			raise RuntimeError("ERROR: Cannot set both job_number and job_name")
	if job_name is not None:
		params['name'] = job_name
		if job_number is not None:
			raise RuntimeError("ERROR: Cannot set both job_number and job_name")

	return params

def get(endpoint, params):
	url = API_URL + "/" + endpoint

	request = urllib2.Request(url, urllib.urlencode(params))
	response = urllib2.urlopen(request)

	return response

def get_lines(endpoint, params):
	ret = get(endpoint, params)
	return ret.read()

def get_json(endpoint, params):
	try:
		ret = get_lines(endpoint, params)
	except urllib2.HTTPError as e:
		if e.code == 404:
			ret = e.read()
			pass
		else:
			raise

	if ret == "":
		ret = "{}"
	
	return json.loads(ret)

def post_json(endpoint, params):
	url = API_URL + "/" + endpoint

	request = urllib2.Request(url, data=json.dumps(params), headers=HEADER)
	response = urllib2.urlopen(request)

	return json.load(response)

def action(username, apikey, action, job_name=None, job_number=None):
	params = {
		'username': username,
		'apikey': apikey,
		'action': action
	}

	params = set_job_name_or_number(params, job_name, job_number)

	ret = get_json("action", params)

	if ret["status"] != "action requested":
		raise RuntimeError("ERROR: Failed to run action")

def shutdown(username, apikey, job_name=None, job_number=None):
	params = {
		'username': username,
		'apikey': apikey,
	}

	params = set_job_name_or_number(params, job_name, job_number)

	ret = get_json("shutdown", params)

	if ret["status"] != "shutdown requested":
		raise RuntimeError("ERROR: Failed to shutdown job")

def submit(username, apikey, job_desc):
	params = job_desc

	user_params = {
		'username': username,
		'apikey': apikey
	}

	params["user"] = user_params

	ret = post_json("submit", params)

	#TODO: Validate that the post was successful by testing for the name/number of job

	return ret 

def terminate(username, apikey, job_name=None, job_number=None):
	params = {
		'username': username,
		'apikey': apikey
	}

	params = set_job_name_or_number(params, job_name, job_number)

	ret = get_json("terminate", params)

	if ret["status"] != "terminated":
		raise RuntimeError("ERROR: Failed to terminate job")

def connect(username, apikey, job_number=None, job_name=None):
	params = {
		'username': username,
		'apikey': apikey
	}

	params = set_job_name_or_number(params, job_name, job_number)

	ret = get_json("connect", params)

	return ret

def info(username, apikey, job_number=None, job_name=None):
	params = {
		'username': username,
		'apikey': apikey
	}

	params = set_job_name_or_number(params, job_name, job_number)

	ret = get_json("info", params)
	
	if not "url" in ret or not "about" in ret:
		raise RuntimeError("ERROR: Failed to get web service information")

	return ret

def jobs(username, apikey):
	params = {
		'username': username,
		'apikey': apikey
	}

	ret = get_json("jobs", params)
	
	return ret

def output(username, apikey, job_number=None, job_name=None, lines=None):
	params = {
		'username': username,
		'apikey': apikey
	}

	params = set_job_name_or_number(params, job_name, job_number)

	if lines is not None:
		params["lines"] = lines

	ret = get_lines("output", params)

	return ret

def status(username, apikey, job_number=None, job_name=None):
	params = {
		'username': username,
		'apikey': apikey
	}

	params = set_job_name_or_number(params, job_name, job_number)

	ret = get_json("status", params)

	# TODO: validate response

	return ret

def tail(username, apikey, job_number=None, job_name=None, lines=None):
	params = {
		'username': username,
		'apikey': apikey
	}

	params = set_job_name_or_number(params, job_name, job_number)

	if lines is not None:
		params["lines"] = lines

	ret = get_lines("tail", params)

	return ret

def lftp(cmds):
	FNULL = open(os.devnull, 'w')
	lftp_proc = subprocess.Popen(["lftp"], stderr=FNULL, stdin=subprocess.PIPE)
	lftp_proc.communicate(input=cmds)[0]

def upload_testcase(username, apikey, testid):
	# chmod after copy is done to allow profile reports to be overwritten
	cmds = """
open sftp://drop.jarvice.com
user %s %s
rm -rf /data/automated_test/%s
mkdir -p /data/automated_test/%s
cd /data/automated_test/%s
mirror -R -X _xocc_*/ -X .Xil/ .
cache flush
bye
""" % (username, apikey, testid, testid, testid)
	return lftp(cmds)

def download_testcase(username, apikey, testid, job_name, out):
	cmds = """
open sftp://drop.jarvice.com
user %s %s
cd /data/automated_test/%s
mirror %s
mirror %s
cache flush
get /data/NACC-OUTPUT/%s.txt
rm -r /data/automated_test/%s
bye
""" % (username, apikey, testid, out, out, job_name, testid)

	return lftp(cmds)

def submit_testcase(username, apikey, testid, exe, args, type, nae, tt, dp):
	_exe = "automated_test/%s/%s" % (testid, exe)
	_args = ' '.join(args)

	job_desc = {
		"app": nae,
		"checkedout": False,
		"machine": {
			"nodes": "1",
			"type": type
		},
		"application": {
			"geometry": "800x600",
			"staging": False,
			"command": "batch",
			"parameters": {
				"executable": _exe,
				"args": _args,
				"-dp": dp,
				"-tt": tt
			}
		},
		"vault": {
			"readonly": False,
			"password": "",
			"force": False,
			"name": "drop.jarvice.com",
			"objects": []
		}
	}

	job = submit(username, apikey, job_desc)

	return job

description = "Application for Running Jobs on Nimbix"
parser = optparse.OptionParser(description=description)
parser.add_option("--nae", help="Set Nimbix Application Environment to use (Advanced)", type=str, default="xilinx-sdx")
parser.add_option("--type", help="Set Nimbix Node Type to use (nx1, nx2, nx3, nx4)", type=str, default="nx3")
parser.add_option("--tt", help="Enable timeline trace", action="store_true", default=False)
parser.add_option("--dp", help="Enable device profiling", action="store_true", default=False)
parser.add_option("--out", help="Set output directory", type=str, default=".")
parser.add_option("--queue_timeout", help="How long to wait for job to run while in queue (minutes)", type=int, default=60)
parser.add_option("--exe_timeout", help="How long to wait for job to run on the board (minutes)", type=int, default=5)
parser.add_option("-v", "--verbose", help="Print out additional information", action="store_true", default=False)

try:
	opts, remainder = parser.parse_args()
except:
	print "ERROR: Could not parse command line options"
	raise

nae = opts.nae
type = opts.type
tt = opts.tt
dp = opts.dp
out = opts.out

exe_timeout = opts.exe_timeout
queue_timeout = opts.queue_timeout

exe = remainder[0]
args = remainder[1:]

if opts.verbose:
	print "NAE = " + nae
	print "TYPE = " + type
	print "-tt = " + str(tt)
	print "-dp = " + str(dp)
	print "exe = " + exe
	print "args = " + ' '.join(args)

cdir = os.getcwd()
rundate = datetime.datetime.now();

user = pwd.getpwuid(os.getuid())[0]
rand = os.urandom(4).encode('hex')
testid = user + "-" + rundate.strftime("%d%m%y_%S") + "-" + rand

try:
	nimbix_user = os.environ['NIMBIX_USER']
	nimbix_apikey = os.environ['NIMBIX_APIKEY']
except KeyError:
	nimbix_user = ""
	nimbix_apikey = ""

if nimbix_user == "" or nimbix_apikey == "":
	creds_filename = os.path.expanduser("~/.nimbix_creds.json")
	if os.path.isfile(creds_filename):
		creds_fp = open(creds_filename)
		creds = json.load(creds_fp)
		nimbix_user = creds['username']
		nimbix_apikey = creds['api-key']
	else:
		print "ERROR: Must set NIMBIX_USER or have ~/.nimbix_creds.json file"
		sys.exit(-1)

if exe == "":
	print "ERROR: Must define the application executable"
	sys.exit(-1)

print "Uploading Test Case"

upload_testcase(nimbix_user, nimbix_apikey, testid)

print "Upload Complete"
print
print "Testcase: " + exe + " " + ' '.join(args)

# Sleep for 10 seconds before trying to execute as sometimes the uploads are not
# instantaneous
time.sleep(10)

job = submit_testcase(nimbix_user, nimbix_apikey,
                      testid, exe, args,
                      type, nae, tt, dp)

print "Job: " + str(job['name']) + " #" + str(job['number'])


# Wait at maximum 1 hour for test to move to running state
queue_end_time = datetime.datetime.now() + datetime.timedelta(minutes=queue_timeout)

rc = 127
exe_end_time = None
api_fails = 0

while True:
	# Sleep for 1 seconds before checking again
	time.sleep(1)

	try:
		current_status = status(nimbix_user, nimbix_apikey, job_number=job['number'])
	except urllib2.URLError:
		# If we fail to access the url a couple times then pass on the failure.
		if api_fails > 5:
			raise
		else:
			api_fails += 1

		continue

	state = current_status[str(job['number'])]['job_status'];

	#  First time state = RUNNING set timeout
	if state == 'RUNNING':
		if not exe_end_time:
			exe_end_time = datetime.datetime.now() + datetime.timedelta(minutes=exe_timeout)

		# Test Timeout
		if datetime.datetime.now() > exe_end_time:
			terminate(nimbix_user, nimbix_apikey,
			          job_number=job['number'])
			rc = 140
			break

	# Infrastructure Failure or invalid arguments
	if state == 'ENDED':
		raise RuntimeError("ERROR: Test entered ENDED state")

	current_time = datetime.datetime.now()

	# Infrastructure Failure if queue timeout exceeded
	if current_time > queue_end_time:
		terminate(nimbix_user, nimbix_apikey,
		                  job_number=job['number'])
		raise RuntimeError("ERROR: Queue timeout exceeded")

	# Need to determine a way to get better error status from Nimbix
	if state == 'COMPLETED':
		rc = 0
		break

	# Need to determine a way to get better error status from Nimbix
	if state == 'COMPLETED WITH ERROR':
		rc = 1
		break

if rc == 0:
	rc_str = "Successfully"
else:
	rc_str = "with Errors"

print "Job Completed %s" % rc_str

print "Downloading Test Case"

download_testcase(nimbix_user, nimbix_apikey, testid, job['name'], out)

output = output(nimbix_user, nimbix_apikey,
                        job_number=job['number'])
print "Download Complete"
print
print output

sys.exit(rc)

