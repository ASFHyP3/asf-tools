#################################
#   ASF ArcGIS Toolbox          #
#   Heidi Kristenson            #
#   Alaska Satellite Facility   #
#   18 September 2020           #
#################################
import os
import sys

import arcpy  # noqa import neccesary but arcpy only available in arcgis environment

sys.path.append(os.path.dirname(__file__))

from unzip import UnzipFiles
from scale_conversion import ScaleConversion
from reclassify_rtc import ReclassifyRTC
from log_diff import LogDiff
from rgb_decomp import RGBDecomp


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the .pyt file)."""
        self.label = "ASF Tools"
        self.alias = "ASF Tools"

        # List of tool classes associated with this toolbox
        self.tools = [UnzipFiles, ScaleConversion, ReclassifyRTC, LogDiff, RGBDecomp]
