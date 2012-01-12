host         = 'localhost'
database     = 'optimize'
runuser      = 'run'
runpassword  = 'g0tm'
viewuser     = 'jorn'
viewpassword = '1r3z2g6$'
adminuser     = None
adminpassword = None
defaultfile   = None

admin  = 0
select = 1
insert = 2

import getpass
import MySQLdb

def connect(task=None):
    # Set default user naem/password based on chosen task.
    username,password = None,None
    if task==admin:
        username,password = adminuser,adminpassword
    elif task==select:
        username,password = viewuser,viewpassword
    elif task==insert:
        username,password = runuser,viewpassword

    # If we do not have a user name yet, and we do not have a default file either, ask the user interactively.
    if username is None and defaultfile is None:
        username = 'root'
        print 'Connecting to MySQL server %s.' % host
        username = raw_input('User name [root]: ')
        if not username: username = 'root'

    # If we do not have a passsword yet, and we do not have a default file either, ask the user interactively.
    if password is None and defaultfile is None:
        password = getpass.getpass('Password for user %s: ' % username)
        
    # Connect to database
    kwargs = {}
    if host     is not None: kwargs['host']   = host
    if database is not None: kwargs['db']     = database
    if username is not None: kwargs['user']   = username
    if password is not None: kwargs['passwd'] = password
    if defaultfile is not None: kwargs['read_default_file'] = defaultfile
    db = MySQLdb.connect(**kwargs)
    
    return db
