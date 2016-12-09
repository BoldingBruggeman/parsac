#!/usr/bin/env python

# Import from standard Python library
import sys, os.path, glob, shutil, subprocess

# Importing Scientific guarantees that the path to Scientific_netcdf is added to the Python path.
import Scientific

# Import our own custom modules
import run

# Shared settings
excludes = ['pylab','Tkconstants','Tkinter','tcl']
datafiles = [('scenario',tuple(glob.glob('%s/*' % run.job.controller.scenariodir))),
             ('',(run.job.controller.exe,)),
             ('obs',tuple(run.job.getObservationPaths()))]
targetdir = 'dist'

tasks = ('cx_Freeze','bbfreeze','py2exe','PyInstaller')

if len(sys.argv)<=1:
    print 'Must specify task to perform (%s) as first argument.' % ', '.join(tasks)
    sys.exit(2)

task = sys.argv[1]
if task not in tasks:
    print 'Unknown task "%s" specified.' % task
    sys.exit(2)

if task=='cx_Freeze':
    # Use cx_Freeze to freeze the client.
    
    # Get path to Scientific_netcdf module
    sppath = os.path.join(os.path.dirname(Scientific.__file__),sys.platform)

    # Run FreezePython
    proc = subprocess.Popen(['FreezePython','--target-dir='+targetdir,'--include-path='+sppath,'--include-modules=encodings.ascii,encodings.latin_1,encodings.utf_8','--exclude-modules='+','.join(excludes),'run.py'])
    proc.wait()
elif task=='bbfreeze':
    from bbfreeze import Freezer
    f = Freezer(targetdir, excludes=excludes)
    f.include_py = False
    f.addScript("run.py")
    f()    # starts the freezing process	
elif task=='PyInstaller':
    sys.path.append('/home/jorn/pyinstaller-1.3')
    import Build
    specnm = 'run'
    Build.SPECPATH = os.getcwd()
    Build.WARNFILE = os.path.join(Build.SPECPATH, 'warn%s.txt' % specnm)
    Build.BUILDPATH = os.path.join(Build.SPECPATH, 'build%s' % specnm)
    if not os.path.exists(Build.BUILDPATH):
        os.mkdir(Build.BUILDPATH)
    a = Build.Analysis([os.path.join(Build.HOMEPATH,'support/_mountzlib.py'), os.path.join(Build.HOMEPATH,'support/useUnicode.py'), os.path.abspath('./run.py')],
             pathex=['/home/jorn/pyinstaller-1.3'])
    pyz = Build.PYZ(a.pure - [(mn,'','') for mn in excludes])
    exe = Build.EXE(pyz,
          a.scripts,
          exclude_binaries=1,
          name='buildrun/run',
          debug=False,
          strip=False,
          upx=False,
          console=1 )
    coll = Build.COLLECT( exe,
               a.binaries,
               strip=False,
               upx=False,
               name='dist')
else:
    # Use py2exe to freeze the client.
    
    from distutils.core import setup
    import py2exe

    # Add path to MSVC 8 CRT, because py2exe does not use WinSxS system.
    sys.path += ['.\\Microsoft.VC80.CRT']

    # Add files of MS Visual C runtime
    datafiles.append(('',(r'C:\WINDOWS\system32\MSVCP71.dll', r'.\Microsoft.VC80.CRT\MSVCR80.dll', r'.\Microsoft.VC80.CRT\Microsoft.VC80.CRT.manifest')))

    setup(console=['run.py'],
          data_files=datafiles,
          options={'py2exe':{'excludes':excludes}}
    )

if task=='cx_Freeze' or task=='bbfreeze' or task=='PyInstaller':
    # Copy data files manually
    for (dirname,files) in datafiles:
        fulldir = os.path.join(targetdir,dirname)
        print 'Copying %i files to %s...' % (len(files),fulldir)
        if not os.path.isdir(fulldir): os.mkdir(fulldir)
        for f in files: shutil.copy(f,fulldir)
	
