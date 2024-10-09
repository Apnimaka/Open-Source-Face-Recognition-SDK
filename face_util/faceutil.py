import ctypes, ctypes.util
from numpy.ctypeslib import ndpointer
import sys
import os

dll_path = os.path.abspath(os.path.dirname(__file__)) + '/C/face_util.dll'
face_util_engine = ctypes.CDLL(dll_path)

align_vertical = face_util_engine.align_vertical
align_vertical.argtypes = [ndpointer(ctypes.c_ubyte, flags='C_CONTIGUOUS'), ctypes.c_int32, ctypes.c_int32, ndpointer(ctypes.c_ubyte, flags='C_CONTIGUOUS'), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float, ctypes.c_float]
align_vertical.restype = ctypes.c_int32

getPose = face_util_engine.getPose
getPose.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
getPose.restype = ctypes.c_int32

getPose68 = face_util_engine.getPose68
getPose68.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
getPose68.restype = ctypes.c_int32
