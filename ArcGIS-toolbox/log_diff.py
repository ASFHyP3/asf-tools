import os

import arcpy  # noqa import neccesary but arcpy only available in arcgis environment


class LogDiff(object):
    def __init__(self):

        """Calculates the Log Difference between two RTC products"""
        self.label = "Calculate Log Difference"
        self.description = "This tool calculates the log difference between two RTC products."
        self.canRunInBackground = True

    def getParameterInfo(self):
        """Define parameter definitions"""

        # First parameter: most recent RTC file
        date2 = arcpy.Parameter(
            name="date2",
            displayName="Comparison raster (i.e. most recent SAR acquisition)",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Input")

        # Second parameter: oldest RTC file
        date1 = arcpy.Parameter(
            name="date1",
            displayName="Reference raster (i.e. oldest SAR acquisition)",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Input")

        # Third parameter: output path for log difference file
        outdir = arcpy.Parameter(
            name="outdir",
            displayName="Destination folder for output file",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input")

        # Fourth parameter: output file name for log difference file
        outname = arcpy.Parameter(
            name="outname",
            displayName="File name for output log difference file (including valid raster file extension)",
            datatype="GPString",
            parameterType="Required",
            direction="Input")

        # Fifth parameter: select if output is added to the map
        outYN = arcpy.Parameter(
            name="outYN",
            displayName="Add output to map",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input")

        outYN.value = "true"

        # Sixth parameter: output layer to add to project
        outlayer = arcpy.Parameter(
            name="outlayer",
            displayName="Derived output for final product raster",
            datatype="GPRasterLayer",
            parameterType="Derived",
            direction="Output")

        params = [date2, date1, outdir, outname, outYN, outlayer]
        return params

    def isLicensed(self):
        """This tool requires the Spatial Analyst Extension"""
        arcpy.AddMessage("Checking Spatial Analyst Extension status...")
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                raise Exception
            else:
                arcpy.AddMessage("Spatial Analyst Extension is available.")
                if arcpy.CheckOutExtension("Spatial") == "CheckedOut":
                    arcpy.AddMessage("Spatial Analyst Extension is checked out and ready for use.")
                elif arcpy.CheckOutExtension("Spatial") == "NotInitialized":
                    arcpy.CheckOutExtension("Spatial")
                    arcpy.AddMessage("Spatial Analyst Extension has been checked out.")
                else:
                    arcpy.AddMessage("Spatial Analyst Extension is not available for use.")
        except Exception:
            arcpy.AddMessage(
                "Spatial Analyst extension is not available for use. "
                "Check your licensing to make sure you have access to this extension.")
            return False

        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""

        # Set the default value for outdir to be the directory of the input most-recent raster
        if parameters[0].value and parameters[1].value:
            workspace = os.path.dirname(parameters[0].value.value)
            if not parameters[2].altered:
                parameters[2].value = workspace

        # Set the default value for outname to be a combination of the input base filenames with a LogDiff tag
        if parameters[0].value and parameters[1].value:
            d2base = os.path.splitext(os.path.basename(parameters[0].value.value))[0]
            d1base = os.path.splitext(os.path.basename(parameters[1].value.value))[0]
            outflnm = str(d1base + '_' + d2base + '_LogDiff.tif')
            if not parameters[3].altered:
                parameters[3].value = outflnm

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""

        # Check licensing
        self.isLicensed()

        # Define parameters
        date2 = parameters[0].valueAsText
        date1 = parameters[1].valueAsText
        outdir = parameters[2].valueAsText
        outname = parameters[3].valueAsText
        outYN = parameters[4].valueAsText

        arcpy.AddMessage("Parameters accepted. Generating Log Difference file %s..." % outname)

        # Run the code to calculate the log difference
        outLogDiff = str(outdir + '\\' + outname)
        outLog10 = arcpy.sa.Log10(arcpy.sa.Divide(date2, date1))
        outLog10.save(outLogDiff)

        # Indicate process is complete
        arcpy.AddMessage("Log Difference raster %s generated." % outname)

        # Add the output product to the map
        if outYN == "true":
            dispname = os.path.splitext(outname)[0]
            arcpy.MakeRasterLayer_management(outLogDiff, dispname)
            arcpy.SetParameterAsText(5, dispname)
            arcpy.AddMessage("Added Log Difference raster layer to map display.")
        else:
            arcpy.AddMessage(
                "Option to add output layer to map was not selected. "
                "Output can be added manually if desired: %s" % outLogDiff)

        # Check In Spatial Analyst Extension
        status = arcpy.CheckInExtension("Spatial")
        messages.addMessage("The Spatial Analyst Extension is in %s status." % status)

        return
