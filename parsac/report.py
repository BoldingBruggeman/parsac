from __future__ import print_function
try:
    input = raw_input
except NameError:
    pass
import sys
import os.path
import threading
import time
import xml.etree.ElementTree

import numpy

from .job import shared
from  . import transport

def fromConfigurationFile(path, description, allowedtransports=None, interactive=True):
    tree = xml.etree.ElementTree.parse(path)

    job_id = os.path.splitext(os.path.basename(path))[0]
    root_dir = os.path.dirname(path)

    # Parse transports section
    if isinstance(allowedtransports, (str, u''.__class__)):
        allowedtransports = (allowedtransports,)
    transports = []
    for itransport, element in enumerate(tree.findall('transports/transport')):
        with shared.XMLAttributes(element, 'transport %i' % (itransport+1)) as att:
            transport_type = att.get('type')
            if allowedtransports is not None and transport_type not in allowedtransports:
                continue
            curtransport = transport.getClass(transport_type).fromXML(att, job_id=job_id, root_dir=root_dir)
        transports.append(curtransport)

    return Reporter(job_id, description, transports, interactive=interactive)

class Reporter:
    def __init__(self, jobid, description, transports=None, interactive=True, separate_thread=True):

        self.jobid = jobid
        self.description = description
        self.separate_thread = separate_thread

        # Check transports
        assert transports is not None, 'One or more transports must be specified.'
        validtp = []
        for transport in transports:
            if transport.available():
                validtp.append(transport)
            else:
                print('Transport %s is not available.' % str(transport))
        if not validtp:
            print('No transport available; exiting...')
            sys.exit(1)
        self.transports = tuple(validtp)

        # Last working transport (to be set at run time)
        self.lasttransport = None

        # Identifier for current run
        self.runid = None

        # Queue with results yet to be reported.
        self.resultqueue = []
        self.allowedreportfailcount = None
        self.allowedreportfailperiod = 3600   # in seconds
        self.timebetweenreports = 60 # in seconds
        self.reportfailcount = 0
        self.lastreporttime = time.time()
        self.reportedcount = 0
        self.nexttransportreset = 30
        self.queuelock = None
        self.reportingthread = None

        # Whether to allow for interaction with user (via e.g. raw_input)
        self.interactive = interactive

    def reportRunStart(self):
        runid = None
        for transport in self.transports:
            if not transport.available(): continue
            try:
                runid = transport.initialize(self.jobid, self.description)
            except Exception as e:
                print('Failed to initialize run over %s.\nReason: %s' % (str(transport), str(e)))
                runid = None
            if runid is not None:
                print('Successfully initialized run over %s.\nRun identifier = %i' % (str(transport), runid))
                self.lasttransport = transport
                break

        if runid is None:
            print('Unable to initialize run. Exiting...')
            sys.exit(1)

        self.runid = runid

    def createReportingThread(self):
        self.queuelock = threading.Lock()
        if self.separate_thread:
            self.reportingthread = ReportingThread(self)
            self.reportingthread.start()

    def reportResult(self, values, result, error=None):
        if self.queuelock is None:
            self.createReportingThread()

        if isinstance(result, tuple):
            lnlikelihood, extra_outputs = result
        else:
            lnlikelihood, extra_outputs = result, None

        if not numpy.isfinite(lnlikelihood):
            lnlikelihood = None

        # Append result to queue
        self.queuelock.acquire()
        self.resultqueue.append((values, lnlikelihood, extra_outputs))
        self.queuelock.release()

        if not self.separate_thread:
            self.flushResultQueue()

    def flushResultQueue(self,maxbatchsize=100):
        # Report the start of the run, if that was not done before.
        if self.runid is None:
            self.reportRunStart()

        while 1:
            # Take current results from the queue.
            self.queuelock.acquire()
            batch = self.resultqueue[:maxbatchsize]
            del self.resultqueue[:maxbatchsize]
            self.queuelock.release()

            # If there are no results to report, do nothing.
            if len(batch) == 0:
                return

            # Reorder transports, prioritizing last working transport
            # Once in a while we retry the different transports starting from the top.
            curtransports = []
            if self.reportedcount < self.nexttransportreset:
                if self.lasttransport is not None: curtransports.append(self.lasttransport)
            else:
                self.nexttransportreset += 30
                self.lasttransport = None
            for transport in self.transports:
                if self.lasttransport is None or transport is not self.lasttransport:
                    curtransports.append(transport)

            # Try to report the results
            for transport in curtransports:
                success = True
                try:
                    transport.reportResults(self.runid, batch, timeout=5)
                except Exception as e:
                    print('Unable to report result(s) over %s. Reason:\n%s' % (str(transport), str(e)))
                    success = False
                if success:
                    print('Successfully delivered %i result(s) over %s.' % (len(batch), str(transport)))
                    self.lasttransport = transport
                    break

            if success:
                # Register success and continue to report any remaining results.
                self.reportedcount += len(batch)
                self.reportfailcount = 0
                self.lastreporttime = time.time()
                continue

            # If we arrived here, reporting failed.
            self.reportfailcount += 1
            print('Unable to report %i result(s). Last report was sent %.0f s ago.' % (len(batch), time.time()-self.lastreporttime))

            # Put unreported results back in queue
            self.queuelock.acquire()
            batch += self.resultqueue
            self.resultqueue = batch
            self.queuelock.release()

            # If interaction with user is not allowed, leave the result in the queue and return.
            if not self.interactive:
                return

            # Check if the report failure tolerances (count and period) have been exceeded.
            exceeded = False
            if self.allowedreportfailcount is not None and self.reportfailcount > self.allowedreportfailcount:
                print('Maximum number of reporting failures (%i) exceeded.' % self.allowedreportfailcount)
                exceeded = True
            elif self.allowedreportfailperiod is not None and time.time() > (self.lastreporttime+self.allowedreportfailperiod):
                print('Maximum period of reporting failure (%i s) exceeded.' % self.allowedreportfailperiod)
                exceeded = True

            # If the report failure tolerance has been exceeded, ask the user whether to continue.
            if exceeded:
                resp = None
                while resp not in ('y', 'n'):
                    resp = input('To report results, connectivity to the server should be restored. Continue for now (y/n)? ')
                if resp == 'n':
                    sys.exit(1)
                self.reportfailcount = 0
                self.lastreporttime = time.time()

            # We will tolerate this failure (the server-side script may be unavailable temporarily)
            print('Queuing current result for later reporting.')
            return

    def finalize(self):
        if self.reportingthread and self.reportingthread.is_alive():
            self.reportingthread.exit_event.set()
            print('Waiting for reporting thread to shut down...')
            self.reportingthread.join(self.timebetweenreports*2)
            if self.reportingthread.is_alive():
                print('WARNING: failed to shut down reporting thread.')
        self.reportingthread = None
        self.runid = None
        self.queuelock = None

class ReportingThread(threading.Thread):
    def __init__(self, job):
        threading.Thread.__init__(self)
        self.exit_event = threading.Event()
        self.job = job

    def run(self):
        done = False
        while not done:
            done = self.exit_event.wait(self.job.timebetweenreports)
            self.job.flushResultQueue()
