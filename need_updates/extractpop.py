#!/usr/bin/env python

# Import from standard Python library
import sys,optparse

# Import third-party modules
import numpy

# Import custom modules
import mysqlinfo

parser = optparse.OptionParser()
(options, args) = parser.parse_args()

assert len(args)>1, 'First argument must be the run identifier, second argument the file to save to.'

db = mysqlinfo.connect(mysqlinfo.select)

# Retrieve all results
c = db.cursor()
query = "SELECT `id`,`parameters` FROM `results` WHERE `run`=%i" % int(args[0])
c.execute(query)
history = []
for resid,strpars in c:
    parameters = map(float,strpars.split(';'))
    history.append(parameters)
db.close()
print 'Found %i results.' % (len(history),)

# Stop if no results were found
if len(history)==0: sys.exit(0)

# Convert results into numpy array
res = numpy.zeros((len(history),len(history[0])))
for i,v in enumerate(history):
    res[i,:] = v

res.dump(args[1])
