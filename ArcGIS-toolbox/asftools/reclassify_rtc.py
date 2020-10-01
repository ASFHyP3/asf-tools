import os

import arcpy  # noqa import neccesary but arcpy only available in arcgis environment


class ReclassifyRTC(object):
    def __init__(self):

        """Reclassifies Raster to apply pixel value of 1 to pixels with original values below a threshold"""
        self.label = "Reclassify RTC"
        self.description = "This tool generates a reclassifed raster based on a threshold value."
        self.canRunInBackground = True

    def getParameterInfo(self):
        """Define parameter definitions"""

        # First parameter: input RTC file to be reclassified
        inRTC = arcpy.Parameter(
            name="inRTC",
            displayName="Raster to be reclassifed",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Input")

        # Second parameter: output path for reclassified raster
        rc_outpath = arcpy.Parameter(
            name="rc_outpath",
            displayName="Destination folder for reclassified raster",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input")

        # Third parameter: output file name for reclassified raster
        rc_outname = arcpy.Parameter(
            name="rc_outname",
            displayName="File name for reclassified raster (must include valid raster file extension)",
            datatype="GPString",
            parameterType="Required",
            direction="Input")

        # Fourth parameter: threshold value
        thresh = arcpy.Parameter(
            name="thresh",
            displayName="Threshold value for reclassification",
            datatype="GPDouble",
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

        params = [inRTC, rc_outpath, rc_outname, thresh, outYN, outlayer]
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
                "Spatial Analyst extension is not available for use. Check your licensing to make sure you have "
                "access to this extension.")
            return False

        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""

        # Set the default value for rc_outpath to be the directory of the input raster
        if parameters[0].value:
            workspaceR = os.path.dirname(parameters[0].value.value)
            if not parameters[1].altered:
                parameters[1].value = workspaceR

        # Set the default value for rc_outname to be the basename of the input raster with a Reclass tag
        if parameters[0].value:
            if not parameters[2].altered:
                outnmR = os.path.splitext(os.path.basename(parameters[0].value.value))[0] + "_Reclass.tif"
                parameters[2].value = outnmR

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
        inRTC = parameters[0].valueAsText
        rc_outpath = parameters[1].valueAsText
        rc_outname = parameters[2].valueAsText
        thresh = parameters[3].valueAsText
        outYN = parameters[4].valueAsText

        # Run the code to reclassify the image
        rcname = str(rc_outpath + '\\' + rc_outname)
        values = "-1000.000000 %s 1;%s 1000.000000 NODATA" % (thresh, thresh)
        arcpy.gp.Reclassify_sa(inRTC, "VALUE", values, rcname, "DATA")

        # Indicate process is complete
        txt_msg3 = "Reclassified raster generated for %s." % (inRTC)
        messages.addMessage(txt_msg3)

        # Add the output product to the map
        if outYN == "true":
            dispname = os.path.splitext(rc_outname)[0]
            arcpy.MakeRasterLayer_management(rcname, dispname)
            arcpy.SetParameterAsText(5, dispname)
            arcpy.AddMessage("Added Reclassified RTC raster layer to map display.")
        else:
            arcpy.AddMessage(
                "Option to add output layer to map was not selected. "
                "Output can be added manually if desired: %s" % rcname)

        # Check In Spatial Analyst Extension
        status = arcpy.CheckInExtension("Spatial")
        arcpy.AddMessage("The Spatial Analyst Extension is in %s status." % status)

        return
