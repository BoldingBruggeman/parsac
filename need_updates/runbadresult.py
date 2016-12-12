#!/usr/bin/env python

import sys,math,optparse

import acpy.run
import acpy.gotmcontroller
import mysqlinfo

parser = optparse.OptionParser()
parser.add_option('-i', '--index', type='int', help='index of bad result (>=0)')
parser.set_defaults(index=0)
(options, args) = parser.parse_args()

if len(args)<1:
    print 'No job identifier provided.'
    sys.exit(2)
jobid = int(args[0])

print 'Looking for bad result %i of job %i.' % (options.index,jobid)

# Connect to database and retrieve best parameter set.
db = mysqlinfo.connect(mysqlinfo.select)
c = db.cursor()
c.execute("SELECT `parameters`,`lnlikelihood` FROM `runs`,`results` WHERE (`runs`.`id`=`results`.`run` AND `runs`.`job`=%i AND `lnlikelihood` IS NULL) LIMIT %i,1" % (jobid,options.index))
parameters = None
for strpars,lnlikelihood in c:
    parameters = map(float,strpars.split(';'))
db.close()
if parameters==None:
    print 'No bad results found. Exiting...'
    sys.exit(0)

job = acpy.run.getJob(jobid)

# Show best parameter set
print 'Testing bad parameter set number %i.' % options.index
print 'Problem parameter set:'
for i,val in enumerate(parameters):
    pi = job.controller.parameters[i]
    if pi['logscale']: val = math.pow(10.,val)
    print '   %s = %.6g' % (pi['name'],val)

# Initialize the GOTM controller.
job.controller.initialize()

# Run and retrieve results.
nc = job.controller.run(parameters,showoutput=True)
