import sys
from us_visa.logger import logging
from us_visa.exception import USVisaException

# logging.info("Welcome to our custom log")
try:
    a = 2/0
except Exception as e:
    raise USVisaException(e, sys)
