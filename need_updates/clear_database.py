#!/usr/bin/env python

import optparse,sys

import mysqlinfo

parser = optparse.OptionParser()
parser.add_option('-r', '--run', type='int', help='run identifier')
parser.add_option('-j', '--job', type='int', help='job identifier')
parser.add_option('--emptyrows', action='store_true', help='delete empty rows only')
parser.set_defaults(emptyrows=False)
(options, args) = parser.parse_args()

db = mysqlinfo.connect(mysqlinfo.admin)

c = db.cursor()

addedwhere = ''
selection = 'all'
if options.emptyrows:
    addedwhere = ' AND (`parameters`=\'\' OR `parameters`=NULL)'
    selection = 'empty'

def delRun(run):
    print 'Removing %s records for run %i from the database...' % (selection,run)
    c.execute('DELETE FROM `results` WHERE (`run`=%i%s);' % (run,addedwhere))
    print '%i records removed from "results" table.' % db.affected_rows()
    if not addedwhere:
        c.execute('DELETE FROM `runs` WHERE `id`=%i;' % (run,))
        print '%i records removed from "runs" table.' % db.affected_rows()

if options.run is not None:
    delRun(options.run)
elif options.job is not None:
    query = "SELECT `id` FROM `runs` WHERE `job`='%i'" % options.job
    c.execute(query)
    for run, in c:
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
