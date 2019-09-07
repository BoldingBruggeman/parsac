# Import from standard Python library
from __future__ import print_function
import socket
import os

# Import third-party modules
import numpy

# Import custom modules
from .. import job
from .. import report
from .. import transport

from . import plot
from . import plotbest
from . import animate_2d
from . import summary

class Result(object):
    def __init__(self, xml_path, simulationdir=None):
        self.job = job.fromConfigurationFile(xml_path, simulationdir=simulationdir)

        reporter = report.fromConfigurationFile(xml_path, '')
        for tp in reporter.transports:
            if isinstance(tp, transport.MySQL):
                import mysqlinfo
                self.db = mysqlinfo.connect(mysqlinfo.select)
            elif isinstance(tp, transport.SQLite):
                import sqlite3
                if not os.path.isfile(tp.path):
                    raise Exception('SQLite database %s does not exist.' % tp.path)
                self.db = sqlite3.connect(tp.path)

    def get_sources(self):
        # Build map from run identifier to source machine
        cursor = self.db.cursor()
        query = "SELECT `id`,`source`,`description` FROM `runs` WHERE `job`='%s'" % self.job.id
        cursor.execute(query)
        run2source = {}
        source2fqn = {}
        for run, source, description in cursor:
            # Chop of run@ string that is prepended if results arrive via MySQL
            if source.startswith('run@'):
                source = source[4:]

            # Try to resolve IP address, to get the host name.
            if source not in source2fqn:
                try:
                    fqn, aliaslist, ipaddrlist = socket.gethostbyaddr(source)
                except:
                    fqn = source
                source2fqn[source] = fqn
            else:
                fqn = source2fqn[source]

            # Link run identifier to source machine
            run2source[run] = source
        cursor.close()

        return run2source

    def get(self, limit=-1, constraints={}, run_id=None, groupby=None, lnlrange=None):
        assert groupby in ('source', 'run', None)

        parconstraints = []
        parnames = self.job.getParameterNames()
        for name, (minval, maxval) in constraints.items():
            minval, maxval = float(minval), float(maxval)
            i = parnames.index(name)
            parconstraints.append((i, minval, maxval))
        npar = len(parnames)

        cursor = self.db.cursor()
        runcrit = '`runs`.`id`'
        if run_id is not None:
            runcrit = '%i' % run_id

        lnl_condition = ''
        if lnlrange is not None:
            print('Finding maximum of fitness function...', end='')
            query = "SELECT DISTINCT MAX(`lnlikelihood`) FROM `runs`,`results` WHERE `results`.`run`=%s AND `runs`.`job`='%s'" % (runcrit, self.job.id)
            cursor.execute(query)
            maxlnl, = cursor.fetchone()
            print(maxlnl)
            lnl_condition = ' AND `lnlikelihood`>%s' % (maxlnl+lnlrange)

        # Retrieve all results
        print('Retrieving results...')
        groupcol = 'NULL' if groupby is None else '`%s`' % groupby
        query = "SELECT DISTINCT `results`.`id`,`parameters`,`lnlikelihood`,%s FROM `runs`,`results` WHERE `results`.`run`=%s AND `runs`.`job`='%s'%s" % (groupcol, runcrit, self.job.id, lnl_condition)
        if limit != -1:
            query += ' LIMIT %i' % limit
        cursor.execute(query)
        history = []
        source2history = {}
        badcount = 0
        i = 1
        for rowid, strpars, lnlikelihood, group in cursor:
            if lnlikelihood is None:
                badcount += 1
            else:
                valid = True
                try:
                    parameters = numpy.array(strpars.split(';'), dtype=float)
                except ValueError:
                    print('Row %i: cannot parse "%s".' % (rowid, strpars))
                    valid = False
                if valid and len(parameters) != npar:
                    print('Row %i: Number of parameters (%i) does not match that of run (%i).' % (rowid, len(parameters), npar))
                    valid = False
                if valid:
                    for ipar, minval, maxval in parconstraints:
                        if parameters[ipar] < minval or parameters[ipar] > maxval:
                            valid = False
                            break
                if valid:
                    dat = numpy.empty((npar+1,), dtype=float)
                    dat[:-1] = parameters
                    dat[-1] = lnlikelihood
                    history.append(dat)
                    if group is not None:
                        source2history.setdefault(group, []).append(dat)
            i += 1
        print('Found %i results, of which %i were invalid.' % (len(history)+badcount, badcount))

        # Convert results into numpy arrays
        all_results = numpy.array(history)

        if groupby is None:
            return all_results

        return all_results, dict([(s, numpy.array(h)) for s, h in source2history.items()])

    def get_best(self, rank=1):
        res = self.get()
        indices = res[:, -1].argsort()
        return res[indices[-rank], :-1], res[indices[-rank], -1]

    def count(self):
        cursor = self.db.cursor()
        count = cursor.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        cursor.close()
        return count
