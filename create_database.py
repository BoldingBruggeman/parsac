import MySQLdb,sys
import mysqlinfo

pw = raw_input('Root password for MySQL server %s: ' % mysqlinfo.host)

db = MySQLdb.connect(host=mysqlinfo.host, user='root',passwd=pw)
c = db.cursor()

print 'Connected to MySQL server %s.' % mysqlinfo.host
resp = ''
while resp not in ('y','n'):
    resp = raw_input('Are you sure you want to delete any existing database "%s" and create a new one? (y/n): ' % mysqlinfo.database)
if resp=='n':
    print 'Database creation cancelled.'
    sys.exit(0)

#c.execute('ALTER TABLE `runs` ADD COLUMN `job` INT AFTER `time`;')
#c.execute('ALTER TABLE `results` MODIFY `parameters` VARCHAR(500);')

# Delete and recreate database
c.execute('DROP DATABASE `%s`;' % mysqlinfo.database)
c.execute('CREATE DATABASE `%s`;' % mysqlinfo.database)

# Create users and grant minimum permissions
c.execute('CREATE USER \'%s\'@\'%\' IDENTIFIED BY \'%s\';' % (mysqlinfo.runuser,mysqlinfo.runpassword))
c.execute('CREATE USER \'%s\'@\'localhost\' IDENTIFIED BY \'%s\';' % (mysqlinfo.viewuser,mysqlinfo.viewpassword))
c.execute('GRANT SELECT ON `%s`.* TO \'%s\'@\'localhost\';' % (mysqlinfo.database,mysqlinfo.viewuser))
c.execute('GRANT INSERT ON `%s`.* TO \'%s\'@\'%\';' % (mysqlinfo.database,mysqlinfo.runuser))

# Create tables
c.execute('USE `%s`;' % mysqlinfo.database)
c.execute('CREATE TABLE `runs`    (`id` INT UNSIGNED NOT NULL AUTO_INCREMENT UNIQUE PRIMARY KEY,`source` VARCHAR(50) NOT NULL,`time` DATETIME NOT NULL,`job` INT,`description` TEXT);')
c.execute('CREATE TABLE `results` (`id` INT UNSIGNED NOT NULL AUTO_INCREMENT UNIQUE PRIMARY KEY,`run` INT UNSIGNED NOT NULL,`time` DATETIME NOT NULL,`parameters` VARCHAR(200) NOT NULL,`lnlikelihood` DOUBLE);')

db.close()
