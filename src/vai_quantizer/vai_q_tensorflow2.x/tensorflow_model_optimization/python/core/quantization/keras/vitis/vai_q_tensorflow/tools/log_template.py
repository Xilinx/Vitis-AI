import logging
#LOG_FORMAT = "%(asctime)s %(levelname)s %(pathname)s %(filename)s %(funcName)s:%(lineno)s : %(message)s"
LOG_FORMAT = "%(asctime)s %(levelname)s %(filename)s %(funcName)s:%(lineno)s]  %(message)s"
DATE_FORMAT = "%Y%m/%d/ %H:%M:%S "

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

logging.debug("This is a debug log.")
logging.info("This is a info log.")
logging.warning("This is a warning log.")
logging.error("This is a error log.")
logging.critical("This is a critical log.")
