from logging import *
import sys

# set up package level logger that by default does nothing
logger=getLogger('forcebalance')
logger.addHandler(NullHandler())
logger.setLevel(INFO)

class RawStreamHandler(StreamHandler):
    """Exactly like output.StreamHandler except it does no extra formatting
    before sending logging messages to the stream. This is more compatible with
    how output has been displayed in ForceBalance. Default stream has also been
    changed from stderr to stdout"""
    def __init__(self, stream = sys.stdout):
        super(RawStreamHandler, self).__init__(stream)
    
    def emit(self, record):
        message = record.getMessage()
        self.stream.write(message)
        self.flush()

class RawFileHandler(FileHandler):
    """Exactly like output.FileHandler except it does no extra formatting
    before sending logging messages to the file. This is more compatible with
    how output has been displayed in ForceBalance."""
    def emit(self, record):
        message = record.getMessage()
        self.stream.write(message)
        self.flush()
        
class CleanFileHandler(FileHandler):
    """File handler that does not write terminal escape codes to files. Use this
    when writing to a file that will probably not be viewed in a terminal"""
    def emit(self, record):
        message = record.getMessage()
        message = re.sub("\x1b\[[0-9][0-9]?;?[0-9]?[0-9]?m", "", message)
        self.stream.write(message)
        self.flush()

