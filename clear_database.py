import optparse,sys

import MySQLdb

parser = optparse.OptionParser()
parser.add_option('-r', '--run', type='int', help='run identifier')
parser.add_option('-j', '--job', type='int', help='job identifier')
(options, args) = parser.parse_args()

db = MySQLdb.connect(host='localhost', user='root',passwd='1r3z2g6',db='optimize')

c = db.cursor()

def delRun(run):
    print 'Removing all records for run %i from the database...' % run
    c.execute('DELETE FROM `results` WHERE `run`=%i;' % run)
    print '%i records removed from "results" table.' % db.affected_rows()
    c.execute('DELETE FROM `runs` WHERE `id`=%i;' % run)
    print '%i records removed from "runs" table.' % db.affected_rows()

if options.run is not None:
    delRun(options.run)
elif options.job is not None:
    query = "SELECT `id` FROM `runs` WHERE `job`='%i'" % options.job
    c.execute(query)
    for run in c:
        delRun(run)
else:
    print 'Currently the clearing of the entire database is disabled for safety.'
    sys.exit(1)
    c.execute('DELETE FROM `results`;')
    print '%i records removed from "results" table.' % db.affected_rows()
    c.execute('DELETE FROM `runs`;')
    print '%i records removed from "runs" table.' % db.affected_rows()

db.commit()
db.close()
