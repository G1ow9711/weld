"""Deny data reads during pure prediction; imports/source loading occur before."""
import builtins
import io
import os
from pathlib import Path
import numpy as np


class NoFileReadsDuringPrediction:
    def __init__(self):
        self.attempts = []

    def __enter__(self):
        self.original_builtin, self.original_io, self.original_os = builtins.open, io.open, os.open
        self.original_path = Path.open
        self.original_numpy_load, self.original_numpy_fromfile = np.load,np.fromfile
        def ordinary(original):
            def checked(file, mode="r", *args, **kwargs):
                if not isinstance(file,int) and ("r" in mode or "+" in mode or not any(x in mode for x in "wax")):
                    self.attempts.append(str(file))
                    raise PermissionError("Pure prediction cannot read files: " + str(file))
                return original(file,mode,*args,**kwargs)
            return checked
        def os_checked(file, flags, *args, **kwargs):
            if (flags & (os.O_WRONLY | os.O_RDWR)) != os.O_WRONLY:
                self.attempts.append(str(file))
                raise PermissionError("Pure prediction cannot open data for reading: " + str(file))
            return self.original_os(file,flags,*args,**kwargs)
        builtins.open = ordinary(self.original_builtin)
        io.open = ordinary(self.original_io)
        # Python 3.10's pathlib accessor captured io.open before this scope;
        # patch its public dispatch as well as current builtin/io functions.
        Path.open = ordinary(self.original_path)
        os.open = os_checked
        def numpy_checked(file,*args,**kwargs):
            self.attempts.append(str(file))
            raise PermissionError("Pure prediction cannot read NumPy data: " + str(file))
        np.load,np.fromfile = numpy_checked,numpy_checked
        return self

    def __exit__(self,*error):
        builtins.open, io.open, os.open = self.original_builtin, self.original_io, self.original_os
        Path.open = self.original_path
        np.load,np.fromfile = self.original_numpy_load,self.original_numpy_fromfile
        return False
