"""Existing-data reads denied; exclusive new numeric diagnostic scratch only.

Windows SciPy Qhull creates a read/write temporary message stream. These
files start empty in a newly owned project-local directory and cannot open
existing inputs, labels, CAD or cached predictions. Guard329 stays unchanged.
"""
import os
import tempfile
from pathlib import Path
from .pure_prediction_file_guard_329 import NoFileReadsDuringPrediction


class PureXYZFreshScratchGuard(NoFileReadsDuringPrediction):
    def __init__(self,scratch_directory):
        super().__init__()
        self.scratch_directory = Path(scratch_directory).resolve(strict=True)
        self.exclusive_empty_scratch_creations = []
        if not self.scratch_directory.is_dir() or any(self.scratch_directory.iterdir()):
            raise ValueError("Fresh empty project-owned numeric scratch directory required")

    def __enter__(self):
        self.previous_tempdir = tempfile.tempdir
        # Bypass Windows tempfile's candidate-directory probe, which itself
        # tries O_RDWR. This exact fresh directory was already verified empty.
        tempfile.tempdir = str(self.scratch_directory)
        super().__enter__()
        deny_existing_reads = os.open
        def fresh_numeric_scratch(file,flags,*args,**kwargs):
            path = Path(os.fsdecode(file)).resolve() if not isinstance(file,int) else None
            read_write = (flags & (os.O_RDONLY | os.O_WRONLY | os.O_RDWR)) == os.O_RDWR
            exclusive_create = (flags & (os.O_CREAT | os.O_EXCL)) == (os.O_CREAT | os.O_EXCL)
            if read_write and exclusive_create and path is not None and path.parent == self.scratch_directory:
                # O_EXCL is enforced by the native open. An existing file
                # cannot be acquired, even between path checking and open.
                fd = self.original_os(file,flags,*args,**kwargs)
                self.exclusive_empty_scratch_creations.append(dict(path=str(path),
                    flags=int(flags),exclusive_create=True,existing_data_read=False))
                return fd
            return deny_existing_reads(file,flags,*args,**kwargs)
        os.open = fresh_numeric_scratch
        return self

    def __exit__(self,*error):
        try:
            return super().__exit__(*error)
        finally:
            tempfile.tempdir = self.previous_tempdir
