import unittest, os, sys, re
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
        'pretend':False
        }
    # handle options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "me:pvq", ["metatest","exclude=","pretend", "verbose", "quiet", "headless-config="])

        if args:
            options['test_modules']=['test_' + module for module in args]
        for o, a in opts:
            if o in ("-m", "--metatest"):
                options['test_modules']=['__test__']
            elif o in ("-e", "--exclude"):
                exclude=a.split(',')

                options['test_modules']=[module[:-3] for module in sorted(os.listdir('test'))
                                        if re.match("^test_.*\.py$",module)
                                        and module[5:-3] not in exclude]
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
If no modules are specified, all test modules in test/ are run

Valid options are:
-e, \t\t\tRun all tests except those for listed modules
--exclude=MODULE[,MODULE2[,...]]
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

    warningHandler = CleanFileHandler('test.err','w')
    warningHandler.setLevel(logging.WARNING)
    debugHandler = CleanFileHandler('test.log','w')
    debugHandler.setLevel(logging.DEBUG)
    
    logging.getLogger("test").addHandler(warningHandler)
    logging.getLogger("test").addHandler(debugHandler)

    options['loglevel']=logging.DEBUG
    
    runner=ForceBalanceTestRunner()
    results=runner.run(**options)

    if not results.wasSuccessful():
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        # establish connection with smtp server
        server = smtplib.SMTP(host = headless_options["smtp_server"],
                              port = headless_options["smtp_port"])
        if headless_options.has_key("smtp_tls") and headless_options["smtp_tls"].lower()=="true":
            server.starttls()
        server.login(user = headless_options["smtp_username"],
                     password = headless_options["smtp_password"])

        # prepare message text
        text = "ForceBalance unit test failure. See attached debug log for more details\n"
        text += "Error Output:\n"
        with open('test.err','r') as warnings:
            text += warnings.read()

        log = ''
        with open('test.log','r') as debug:
            log += debug.read()

        # format message headers
        msg = MIMEMultipart()
        msg['From']= headless_options["source"]
        msg['To']= headless_options["destination"]
        msg['Subject']=headless_options["subject"]

        content = MIMEText(text)
        attachment = MIMEText(log)
        attachment.add_header('Content-Disposition', 'attachment', filename="debug.log")
        msg.attach(content)
        msg.attach(attachment)

        # send message
        server.sendmail(headless_options["source"],
                        [ headless_options["destination"] ],
                        msg.as_string())

        # close connection
        server.quit()

def run(options):
    logging.getLogger("test").addHandler(forcebalance.nifty.RawStreamHandler(sys.stderr))
    runner=ForceBalanceTestRunner()
    runner.run(**options)


#### main block ####

options=getOptions()

if options.has_key('headless_config'):
    runHeadless(options)
else:
    run(options)

sys.exit()
