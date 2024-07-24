'''Logging Module'''
import logging

class Logger:
    '''Logger Class'''
    @staticmethod
    def setup_logging(config):
        '''Sets up the logging configuration'''

        logging.basicConfig(
            filename=config['logging']['file'],
            level=config['logging']['level'],
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
