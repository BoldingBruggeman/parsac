import os.path
import json

try:
    import MySQLdb
except ImportError:
    MySQLdb = None

try:
    import sqlite3
except ImportError:
    sqlite3 = None

try:
    import httplib
    import socket
except ImportError:
    httplib = None

try:
    from urllib import parse as urllib_parse
except ImportError:
    import urllib as urllib_parse

def getClass(transport_type):
    if transport_type not in type2class:
        raise Exception('Unknown transport type "%s"."' % transport_type)
    return type2class[transport_type] 

class Transport:
    @classmethod
    def fromXML(cls, att, **kwargs):
        raise NotImplementedError()

    def __init__(self):
        pass

    def available(self):
        return True

    def initialize(self, jobid, description):
        raise NotImplementedError('Method "initialize" must be implemented by derived class.')

    def reportResults(self, runid, results, timeout=5):
        raise NotImplementedError('Method "reportResults" must be implemented by derived class.')

class Dummy(Transport):
    def initialize(self, jobid, description):
        return 1

    def reportResults(self, runid, results, timeout=5):
        pass

class MySQL(Transport):
    @classmethod
    def fromXML(cls, att, **kwargs):
        defaultfile = att.get('defaultfile', required=False)
        return cls(server=att.get('server', required=(defaultfile is None)),
                   user=att.get('user', required=(defaultfile is None)),
                   password=att.get('password', required=(defaultfile is None)),
                   database=att.get('database', required=(defaultfile is None)),
                   defaultfile = defaultfile)

    def __init__(self, server, user, password, database, timeout=30, defaultfile=None):
        Transport.__init__(self)
        self.server   = server
        self.user     = user
        self.password = password
        self.database = database
        self.timeout  = timeout
        self.defaultfile = defaultfile

    def available(self):
        return MySQLdb is not None

    def __str__(self):
        return 'MySQL connection to %s' % self.server

    def connect(self):
        kwargs = {}
        if self.server      is not None: kwargs['host']              = self.server
        if self.user        is not None: kwargs['user']              = self.user
        if self.password    is not None: kwargs['passwd']            = self.password
        if self.database    is not None: kwargs['db']                = self.database
        if self.defaultfile is not None: kwargs['read_default_file'] = self.defaultfile
        if self.timeout     is not None: kwargs['connect_timeout']   = self.timeout 
        return MySQLdb.connect(**kwargs)

    def initialize(self, jobid, description):
        db = self.connect()
        c = db.cursor()
        c.execute("INSERT INTO `runs` (`source`,`time`,`job`,`description`) values(USER(),NOW(),'%i','%s');" % (jobid, MySQLdb.escape_string(description)))
        runid = db.insert_id()
        db.commit()
        db.close()
        return runid

    def reportResults(self, runid, results, timeout=5):
        # Connect to MySQL database and obtain cursor.
        db = self.connect()
        c = db.cursor()

        # Enumerate over results and insert them in the database.
        for values, lnlikelihood, extra_outputs in results:
            strpars = ';'.join('%.12e' % v for v in values)
            if lnlikelihood is None:
                lnlikelihood = 'NULL'
            else:
                lnlikelihood = '\'%.12e\'' % lnlikelihood
            c.execute("INSERT INTO `results` (`run`,`time`,`parameters`,`lnlikelihood`) values(%i,NOW(),'%s',%s);" % (runid, strpars, lnlikelihood))

        # Commit and close database connection.
        db.commit()
        db.close()

class HTTP(Transport):
    @classmethod
    def fromXML(cls, att, **kwargs):
        return cls(server=att.get('server'), path=att.get('path'))

    def __init__(self, server, path):
        Transport.__init__(self)
        self.server = server
        self.path = path

    def available(self):
        return httplib is not None

    def __str__(self):
        return 'HTTP connection to %s' % self.server

    def initialize(self, jobid, description):
        params = {'job':jobid, 'description':description}
        headers = {'Content-type':'application/x-www-form-urlencoded', 'Accept':'text/plain'}
        conn = httplib.HTTPConnection(self.server)
        conn.request('POST', self.path+'startrun.php', urllib_parse.urlencode(params), headers)
        response = conn.getresponse()
        resp = response.read()
        if response.status != httplib.OK:
            raise Exception('Unable to initialize run over HTTP connection.\nResponse:\n%s' % resp)
        runid = int(resp)
        conn.close()
        return runid

    def reportResults(self, runid, results, timeout=5):
        # Create POST parameters and header
        strpars, strlnls = [], []
        for values, lnlikelihood, extra_outputs in results:
            curpars = ';'.join('%.12e' % v for v in values)
            if lnlikelihood is None:
                curlnl = ''
            else:
                curlnl = '%.12e' % lnlikelihood
            strpars.append(curpars)
            strlnls.append(curlnl)
        params = {'run': runid, 'parameters': ','.join(strpars), 'lnlikelihood': ','.join(strlnls)}
        headers = {'Content-type':'application/x-www-form-urlencoded', 'Accept':'text/plain'}

        # Set default socket timeout and remember old value.
        oldtimeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)

        try:
            # Connect and post results to HTTP server.
            conn = httplib.HTTPConnection(self.server)
            conn.request('POST', self.path+'submit.php', urllib_parse.urlencode(params), headers)

            # Interpret server response.
            response = conn.getresponse()
            respstart = response.read(7)
            if response.status != httplib.OK or respstart != 'success':
                raise Exception('Unable to send results over HTTP connection. Server response: '+(respstart+response.read()))

            # Close connection to HTTP server.
            conn.close()
        finally:
            # Restore old default socket timeout
            socket.setdefaulttimeout(oldtimeout)

class SQLite(Transport):
    @classmethod
    def fromXML(cls, att, job_id, root_dir, **kwargs):
        return cls(path=os.path.join(root_dir, att.get('path', default='%s.db' % job_id)))

    def __init__(self, path):
        Transport.__init__(self)
        self.path = path

    def available(self):
        return sqlite3 is not None

    def __str__(self):
        return 'SQLite database %s' % self.path

    def connect(self):
        return sqlite3.connect(self.path)

    def getColumnNames(self, db, tablename):
        c = db.cursor()
        c.execute('PRAGMA table_info(%s);' % tablename)
        return [row[1] for row in c]

    def initialize(self, jobid, description):
        import platform
        create = not os.path.isfile(self.path)
        db = self.connect()
        c = db.cursor()
        if create:
            c.execute('CREATE TABLE `runs`    (`id` INTEGER PRIMARY KEY,`source` VARCHAR(50) NOT NULL,`time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,`job` TEXT NOT NULL,`description` TEXT);')
            c.execute('CREATE TABLE `results` (`id` INTEGER PRIMARY KEY,`run` INTEGER UNSIGNED NOT NULL,`time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,`parameters` TEXT NOT NULL,`lnlikelihood` DOUBLE,`extra_outputs` JSON);')
        else:
            column_names = self.getColumnNames(db, 'results')
            if 'extra_outputs' not in column_names:
                c.execute('ALTER TABLE `results` ADD COLUMN `extra_outputs` JSON;')
        c.execute("INSERT INTO `runs` (`source`,`job`,`description`) values(?,?,?);",(platform.node(), jobid, description))
        runid = c.lastrowid
        db.commit()
        db.close()
        return runid

    def reportResults(self, runid, results, timeout=5):
        # Connect to MySQL database and obtain cursor.
        db = self.connect()
        c = db.cursor()

        # Enumerate over results and insert them in the database.
        for values, lnlikelihood, extra_outputs in results:
            strpars = ';'.join('%.15e' % v for v in values)
            if extra_outputs is not None:
                extra_outputs = json.dumps(extra_outputs)
            c.execute("INSERT INTO `results` (`run`,`parameters`,`lnlikelihood`,`extra_outputs`) values(?,?,?,?);", (runid, strpars, lnlikelihood, extra_outputs))

        # Commit and close database connection.
        db.commit()
        db.close()

type2class = {'mysql': MySQL, 'http': HTTP, 'sqlite': SQLite}
