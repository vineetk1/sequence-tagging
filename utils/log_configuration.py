"""
Vineet Kumar, xyoom.ai
"""

# DEBUG INFO WARN ERROR/EXCEPTION CRITICAL
from typing import Dict, Any


LOG_CONFIG: Dict[str, Any] = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'consoleFormatter': {
            'format':
            '%(levelname)-6s %(filename)s:%(lineno)s:%(funcName)s(): %(message)s',
        },
        'fileFormatter': {
            'format':
            '[%(asctime)s] %(levelname)-6s %(filename)s:%(lineno)s:%(funcName)s(): %(message)s',
        },
    },
    'handlers': {
        'file': {
            'filename': 'ctb_logs',
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'formatter': 'fileFormatter',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'consoleFormatter',
        },
    },
    'loggers': {
        '':
        {  # root logger
            'handlers': ['file', 'console'],
            'level': 'DEBUG',
        },
    },
}
