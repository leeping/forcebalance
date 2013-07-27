import unittest, os, sys, re, shutil, time
import forcebalance
from forcebalance import logging
from __init__ import ForceBalanceTestRunner
import getopt

def getOptions():
    """Parse options passed to forcebalance testing framework"""
    # set some defaults
    exclude = []
    options = {
        'loglevel' : logging.INFO,
        'pretend':False,
        }
    # handle options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "mepvq", ["metatest","exclude","pretend", "verbose", "quiet", "headless-config="])

        # first parse module arguments which determine what modules to test
        exclude = False
        if args:
            for o, a in opts:
                if o in ("-e","--exclude"): exclude = True
            module_args=[]
            for module in args:
                module = module.split('.')
                module_args.append('test_' + module[0])
            if exclude:
                options['test_modules']=[module[:-3] for module in sorted(os.listdir('test'))
                                        if re.match("^test_.*\.py$",module)
                                        and module[:-3] not in module_args]
            else:
                options['test_modules']=module_args

        for o, a in opts:
            if o in ("-m", "--metatest"):
                options['test_modules']=['__test__']
            if o in ("-p", "--pretend"):
                options['pretend']=True
            if o in ("-v", "--verbose"):
                options['loglevel']=logging.DEBUG
            elif o in ("-q", "--quiet"):
                options['loglevel']=logging.WARNING
            if o in ("--headless-config",):
                options['headless_config'] = a
                
    except getopt.GetoptError as err:
        usage()
        sys.exit()
    return options

def usage():
    """Print information on running tests using this script"""
    print """ForceBalance Test Suite
Usage: python test [OPTIONS] [MODULES]
Test suite will load all tests for forcebalance modules provided as arguments.
If no modules are specified, all test modules in test/ are run

Valid options are:
-e, --exclude\t\tRun all tests EXCEPT those listed
-m, --metatest\t\tRun tests on testing framework
-p, --pretend\t\tLoad tests but don't actually run them
-v, --verbose\t\tSet log level to DEBUG, printing additional test information
-q, --quiet\t\tSet log level to WARNING, printing only on failure or error
"""


def runHeadless(options):
    headless_options=dict()
    with open(options["headless_config"], 'r') as f:
        config = f.read().splitlines()
        for line in config:
            line = line.strip()
            if line:
                line = line.split('=')
                headless_options[line[0]]=line[1]

    os.chdir(os.path.dirname(__file__) + "/..")

    class CleanFileHandler(logging.FileHandler):
        """File handler that does not write terminal escape codes to files, which
        makes it easier to do things like send a log via email"""
        def emit(self, record):
            message = record.getMessage()
            message = re.sub("\x1b\[[0-9][0-9]?;?[0-9]?[0-9]?m", "", message)
            self.stream.write(message)
            self.flush()

    os.mkdir('/tmp/forcebalance')
    warningHandler = CleanFileHandler('/tmp/forcebalance/test.err','w')
    warningHandler.setLevel(logging.WARNING)
    logfile = "/tmp/forcebalance/%s.log" % time.strftime('%m%d%y_%H%M%S')
    debugHandler = CleanFileHandler(logfile,'w')
    debugHandler.setLevel(logging.DEBUG)
    
    logging.getLogger("forcebalance.test").addHandler(warningHandler)
    logging.getLogger("forcebalance.test").addHandler(debugHandler)

    options['loglevel']=logging.DEBUG
    
    runner=ForceBalanceTestRunner()
    results=runner.run(**options)

    if headless_options.has_key('enable_smtp')\
    and headless_options['enable_smtp'].lower() in ['true','error']:
        if headless_options['enable_smtp'].lower()=='true' or not results.wasSuccessful():
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # establish connection with smtp server
            server = smtplib.SMTP(host = headless_options["smtp_server"],
                                  port = headless_options["smtp_port"])
            if headless_options.has_key("smtp_tls") and headless_options["smtp_tls"].lower()=="true":
                server.starttls()
            server.login(user = headless_options["smtp_user"],
                         password = headless_options["smtp_password"])

            # prepare message text
            
            text = "ForceBalance unit test suite ran at %s with %d failures, %d errors. " %\
            (time.strftime('%x %X %Z'),len(results.failures), len(results.errors)) 
            text += "See attached debug log for more details\n"
            if not results.wasSuccessful():
                text += "Error Output:\n"
                with open('/tmp/forcebalance/test.err','r') as warnings:
                    text += warnings.read()

            log = ''
            with open(logfile,'r') as debug:
                log += debug.read()

            # format message headers
            msg = MIMEMultipart()
            msg['From']= headless_options["smtp_source"]
            msg['To']= headless_options["smtp_destination"]
            msg['Subject']=headless_options["smtp_subject"]

            content = MIMEText(text)
            attachment = MIMEText(log)
            attachment.add_header('Content-Disposition', 'attachment', filename="debug.log")
            msg.attach(content)
            msg.attach(attachment)

            # send message
            server.sendmail(headless_options["smtp_source"],
                            [ headless_options["smtp_destination"] ],
                            msg.as_string())

            # close connection
            server.quit()
    if headless_options.has_key('log_location'):
        shutil.copy(logfile, headless_options['log_location'])
    shutil.rmtree('/tmp/forcebalance')

def run(options):
    logging.getLogger("forcebalance.test").addHandler(forcebalance.nifty.RawStreamHandler(sys.stderr))
    runner=ForceBalanceTestRunner()
    runner.run(**options)


#### main block ####

options=getOptions()

if options.has_key('headless_config'):
    runHeadless(options)
else:
    run(options)

sys.exit()
