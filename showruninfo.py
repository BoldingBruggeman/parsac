# Import from standard Python library
import sys,optparse

# Import third-party modules
import MySQLdb

# Import custom Python modules
import mysqlinfo

parser = optparse.OptionParser()
(options, args) = parser.parse_args()

if len(args)==0:
    print 'No run identifier specified.'
    sys.exit(1)
runid = int(args[0])

print 'Showing information for run with identifier %i' % runid

# Retrieve run information
db = MySQLdb.connect(host=mysqlinfo.host,user=mysqlinfo.viewuser,passwd=mysqlinfo.viewpassword,db=mysqlinfo.database)
c = db.cursor()
c.execute("SELECT `source`,`time`,`description` FROM `runs` WHERE `id`=%i" % runid)
for (source,time,description) in c:
    print 'Source: %s' % source
    print 'Start time: %s' % time

    if description==None:
        print 'No description was provided.'
        continue
    import pickle
    info = pickle.loads(description)
    
    print 'Parameters:'
    for p in info['parameters']:
        print '   %s/%s/%s, range: %.6g - %.6g, log scale: %s' % (p['namelistfile'],
                                                               p['namelistname'],
                                                               p['name'],
                                                               p['minimum'],
                                                               p['maximum'],
                                                               p['logscale'])
    print 'Observations:'
    for p in info['observations']:
        print '   file: %s, variable: %s, relative fit: %s' % (p['sourcepath'],p['outputvariable'],p['relativefit'])

db.close()

