import os
import sys
import ctypes

# Preload torch's libiomp5md first so fbgemm.dll and MKL both use the compatible runtime
torch_lib = os.path.join(os.path.dirname(__file__), "torch", "lib")
if os.path.isdir(torch_lib):
    try:
        os.add_dll_directory(torch_lib)
    except Exception:
        pass
    iomp5 = os.path.join(torch_lib, "libiomp5md.dll")
    if os.path.isfile(iomp5):
        try:
            ctypes.CDLL(iomp5)
        except Exception:
            pass

# Add Library/bin to PATH and DLL search directory
library_bin = os.path.join(sys.prefix, "Library", "bin")
if os.path.isdir(library_bin):
    try:
        os.add_dll_directory(library_bin)
    except Exception:
        pass
    cur_path = os.environ.get("PATH", "")
    if library_bin not in cur_path:
        os.environ["PATH"] = library_bin + os.pathsep + cur_path

# Ensure GDAL_DATA and PROJ_LIB are configured
gdal_data = os.path.join(sys.prefix, "Library", "share", "gdal")
if os.path.isdir(gdal_data) and "GDAL_DATA" not in os.environ:
    os.environ["GDAL_DATA"] = gdal_data

proj_data = os.path.join(sys.prefix, "Library", "share", "proj")
if os.path.isdir(proj_data) and "PROJ_LIB" not in os.environ:
    os.environ["PROJ_LIB"] = proj_data
