import unittest, os, sys, re, shutil, time
import forcebalance
from __init__ import ForceBalanceTestRunner
import getopt
import argparse

def getOptions():
    """Parse options passed to forcebalance testing framework"""
    # set some defaults
    options = {
        'loglevel' : forcebalance.output.INFO,
        'pretend':False,
        }
    # handle options
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exclude', action='store_true', help="exclude specified modules from test loading")
    parser.add_argument('-p', '--pretend', action='store_true', help="load tests but don't actually run them")
    parser.add_argument('--no-color', action='store_true', help="print test results in black and white (no extra formatting)")
    loglevel = parser.add_mutually_exclusive_group()
    loglevel.add_argument('-v', '--verbose', dest='loglevel', action='store_const', const=forcebalance.output.DEBUG, default=forcebalance.output.INFO)
    loglevel.add_argument('-q', '--quiet', dest='loglevel', action='store_const', const=forcebalance.output.WARNING, default=forcebalance.output.INFO)
    parser.add_argument('--headless-config', type=argparse.FileType('r'), help="run tests in 'headless' mode, using the\
    configuration from the config file provided")
    parser.add_argument('test_modules', metavar="MODULE", nargs='*', default=[], help="module to load tests from")
    
    options = vars(parser.parse_args())
    
    for n, module in enumerate(options['test_modules']):
        options['test_modules'][n] = 'test_' + module.split('.')[0]

    if options['test_modules']:
        if options['exclude']:
            options['test_modules']=[module[:-3] for module in sorted(os.listdir(os.path.dirname(__file__)))
                                if re.match("^test_.*\.py$",module)
                                and module[:-3] not in options['test_modules']]
    else:
        del options['test_modules']
    return options

def runHeadless(options):
    headless_options=dict()
    with options["headless_config"] as f:
        config = f.read().splitlines()
        for line in config:
            line = line.strip()
            if line:
                line = line.split('=')
                headless_options[line[0]]=line[1]

    os.chdir(os.path.dirname(__file__) + "/..")

    if not os.path.exists('/tmp/forcebalance'): os.mkdir('/tmp/forcebalance')
    warningHandler = forcebalance.output.CleanFileHandler('/tmp/forcebalance/test.err','w')
    warningHandler.setLevel(forcebalance.output.WARNING)
    logfile = "/tmp/forcebalance/%s.log" % time.strftime('%m%d%y_%H%M%S')
    debugHandler = forcebalance.output.CleanFileHandler(logfile,'w')
    debugHandler.setLevel(forcebalance.output.DEBUG)
    
    forcebalance.output.getLogger("forcebalance.test").addHandler(warningHandler)
    forcebalance.output.getLogger("forcebalance.test").addHandler(debugHandler)

    options['loglevel']=forcebalance.output.DEBUG
    
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
    if options["no_color"]:
        forcebalance.output.getLogger("forcebalance.test").addHandler(forcebalance.output.CleanStreamHandler(sys.stderr))
    else:
        forcebalance.output.getLogger("forcebalance.test").addHandler(forcebalance.output.RawStreamHandler(sys.stderr))
    runner=ForceBalanceTestRunner()
    runner.run(**options)


#### main block ####

options=getOptions()

if options['headless_config']:
    runHeadless(options)
else:
    run(options)

sys.exit()
