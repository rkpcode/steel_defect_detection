"""
Custom Exception Module for Steel Defect Detection System
"""
import sys

def error_message_detail(error, error_detail: sys):
    """Extract detailed error message with file name and line number"""
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = (
        f"Error occurred in python script: [{file_name}] "
        f"line number: [{exc_tb.tb_lineno}] "
        f"error message: [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    """Custom Exception class with detailed error information"""
    
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
