import sys,math,optparse

# Import third-party modules
import MySQLdb

import client.run,client.gotmcontroller

parser = optparse.OptionParser()
parser.add_option('-j', '--job',   type='int', help='job identifier')
parser.add_option('-i', '--index', type='int', help='index of bad result (>=0)')
parser.set_defaults(job=0,index=0)
(options, args) = parser.parse_args()

print 'Looking for bad result %i of job %i.' % (options.index,options.job)

# Connect to database and retrieve best parameter set.
db = MySQLdb.connect(host='localhost',user='jorn',passwd='1r3z2g6$',db='optimize')
c = db.cursor()
c.execute("SELECT `parameters`,`lnlikelihood` FROM `runs`,`results` WHERE (`runs`.`id`=`results`.`run` AND `runs`.`job`=%i AND `lnlikelihood` IS NULL) LIMIT %i,1" % (options.job,options.index))
parameters = None
for strpars,lnlikelihood in c:
    parameters = map(float,strpars.split(';'))
db.close()
if parameters==None:
    print 'No bad results found. Exiting...'
    sys.exit(0)

# Show best parameter set
print 'Testing bad parameter set number %i.' % options.index
print 'Problem parameter set:'
for i,val in enumerate(parameters):
    pi = client.run.job.controller.parameters[i]
    if pi['logscale']: val = math.pow(10.,val)
    print '   %s = %.6g' % (pi['name'],val)

# Initialize the GOTM controller.
client.run.job.controller.initialize()

# Run and retrieve results.
nc = client.run.job.controller.run(parameters,showoutput=True)
