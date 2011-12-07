import MySQLdb,sys
import mysqlinfo

print 'Database creation disabled for security reasons.'
sys.exit(1)

db = MySQLdb.connect(host=mysqlinfo.host, user='root',passwd='1r3z2g6')
c = db.cursor()

#c.execute('ALTER TABLE `runs` ADD COLUMN `job` INT AFTER `time`;')
#c.execute('ALTER TABLE `results` MODIFY `parameters` VARCHAR(500);')

# Delete and recreate database
c.execute('DROP DATABASE `%s`;' % mysqlinfo.database)
c.execute('CREATE DATABASE `%s`;' % mysqlinfo.database)

# Create users and grant minimum permissions
c.execute('CREATE USER \'%s\'@\'%\' IDENTIFIED BY \'%s\';' % (mysqlinfo.runuser,mysqlinfo.runpassword))
c.execute('CREATE USER \'%s\'@\'localhost\' IDENTIFIED BY \'%s\';' % (mysqlinfo.viewuser,mysqlinfo.viewpassword)
c.execute('GRANT SELECT ON `%s`.* TO \'%s\'@\'localhost\';' % (mysqlinfo.database,mysqlinfo.viewuser))
c.execute('GRANT INSERT ON `%s`.* TO \'%s\'@\'%\';' % (mysqlinfo.database,mysqlinfo.runuser))

# Create tables
c.execute('USE `%s`;' % mysqlinfo.database)
c.execute('CREATE TABLE `runs`    (`id` INT UNSIGNED NOT NULL AUTO_INCREMENT UNIQUE PRIMARY KEY,`source` VARCHAR(50) NOT NULL,`time` DATETIME NOT NULL,`job` INT,`description` TEXT);')
c.execute('CREATE TABLE `results` (`id` INT UNSIGNED NOT NULL AUTO_INCREMENT UNIQUE PRIMARY KEY,`run` INT UNSIGNED NOT NULL,`time` DATETIME NOT NULL,`parameters` VARCHAR(200) NOT NULL,`lnlikelihood` DOUBLE);')

db.close()
