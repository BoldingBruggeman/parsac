import MySQLdb,sys

print 'Database creation disabled for security reasons.'
sys.exit(1)

db = MySQLdb.connect(host='localhost', user='root',passwd='1r3z2g6')
c = db.cursor()

#c.execute('ALTER TABLE `runs` ADD COLUMN `job` INT AFTER `time`;')
#c.execute('ALTER TABLE `results` MODIFY `parameters` VARCHAR(500);')

# Delete and recreate database
c.execute('DROP DATABASE `optimize`;')
c.execute('CREATE DATABASE `optimize`;')

# Create users and grant minimum permissions
c.execute('CREATE USER \'run\'@\'%\' IDENTIFIED BY \'g0tm\';')
c.execute('CREATE USER \'jorn\'@\'localhost\' IDENTIFIED BY \'1r3z2g6$\';')
c.execute('GRANT SELECT ON `optimize`.* TO \'jorn\'@\'localhost\';')
c.execute('GRANT INSERT ON `optimize`.* TO \'run\'@\'%\';')

# Create tables
c.execute('USE `optimize`;')
c.execute('CREATE TABLE `runs`    (`id` INT UNSIGNED NOT NULL AUTO_INCREMENT UNIQUE PRIMARY KEY,`source` VARCHAR(50) NOT NULL,`time` DATETIME NOT NULL,`job` INT,`description` TEXT);')
c.execute('CREATE TABLE `results` (`id` INT UNSIGNED NOT NULL AUTO_INCREMENT UNIQUE PRIMARY KEY,`run` INT UNSIGNED NOT NULL,`time` DATETIME NOT NULL,`parameters` VARCHAR(200) NOT NULL,`lnlikelihood` DOUBLE);')

db.close()
