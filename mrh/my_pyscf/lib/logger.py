from pyscf.lib.logger import *

def select_log_printer (log, tverbose=INFO):
    def noprint(*args, **kwargs):
        return None
    return (noprint, log.error, log.warn, log.note, log.info, log.debug, log.debug1, log.debug2,
            log.debug3, log.debug4, print)[tverbose]



