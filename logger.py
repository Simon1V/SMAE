# -*- coding: utf-8 -*-
from logging.handlers import RotatingFileHandler
import logging 
import sys 

class BLSCLogger:
	def setup_conditional_logger(log_file_path:str, debug_level_console:int, debug_level_file:int, conditional_formatter_for_console:bool=False):
		rotating_file = RotatingFileHandler(log_file_path , mode='a', maxBytes=5 * 1024 * 1024, backupCount=5, encoding=None, delay=0)

		class ConditionalFormatter(logging.Formatter):
			def format(self, record):
				if hasattr(record, 'simple') and record.simple:
					return record.getMessage()
				else:
					return logging.Formatter.format(self, record)

		rotating_file_formatter = ConditionalFormatter('%(asctime)s %(levelname)s - %(message)s')
		rotating_file.setFormatter(rotating_file_formatter)
		rotating_file.setLevel(debug_level_file)
		# The console handler is not conditional, the 'simple' attribute wont work.
		consoleLogger = logging.StreamHandler(sys.stdout)
		if conditional_formatter_for_console == False: 
			consoleLoggerFormatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
		else: 
			consoleLoggerFormatter = ConditionalFormatter('%(asctime)s %(levelname)s - %(message)s')
		consoleLogger.setFormatter(consoleLoggerFormatter)
		consoleLogger.setLevel(debug_level_console)

		root_logger = logging.getLogger()
		root_logger.setLevel(logging.DEBUG)
		root_logger.addHandler(rotating_file)
		root_logger.addHandler(consoleLogger)
		return root_logger 

	def setup_simple_logger(self, name:str, log_file:str, level:int=logging.INFO):
		file_handler = logging.FileHandler(log_file)  
		formatter = logging.Formatter('%(message)s')      
		file_handler.setFormatter(formatter)
		logger = logging.getLogger(name)
		logger.setLevel(level)
		logger.addHandler(file_handler)
		return logger

	def setup_standard_logger(self, name:str, log_file:str, level:int=logging.DEBUG):
		formatter = logging.Formatter('%(levelname)s %(asctime)s - %(message)s')      

		console_handler = logging.StreamHandler(sys.stdout)
		console_handler.setFormatter(formatter)
		console_handler.setLevel(level)

		file_handler = logging.FileHandler(log_file)  
		file_handler.setFormatter(formatter)
		logger = logging.getLogger(name)
		logger.setLevel(level)
		logger.addHandler(file_handler)
		logger.addHandler(console_handler)
		return logger	

	def setup_file_only_logger(self, name:str, log_file_path:str, debug_level_console:int, debug_level_file:int, conditional_formatter_for_console:bool=False):
		rotating_file = RotatingFileHandler(log_file_path , mode='a', maxBytes=5 * 1024 * 1024, backupCount=5, encoding=None, delay=0)
		class ConditionalFormatter(logging.Formatter):
			def format(self, record):
				if hasattr(record, 'simple') and record.simple:
					return record.getMessage()
				else:
					return logging.Formatter.format(self, record)

		rotating_file_formatter = ConditionalFormatter('%(levelname)s %(asctime)s - %(message)s')
		rotating_file.setFormatter(rotating_file_formatter)
		rotating_file.setLevel(debug_level_file)
		# The console handler is not conditional, the 'simple' attribute wont work. 
		root_logger = logging.getLogger()
		root_logger.setLevel(logging.DEBUG)
		root_logger.addHandler(rotating_file)
		return root_logger 
