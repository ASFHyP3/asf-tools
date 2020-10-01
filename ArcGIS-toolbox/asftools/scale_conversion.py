import os

import arcpy  # noqa import neccesary but arcpy only available in arcgis environment


class ScaleConversion(object):
    def __init__(self):

        """Converts RTC products from power to amplitude"""
        self.label = "Scale Conversion (Power, Amplitude, dB)"
        self.description = "This tool converts RTC products from Power or Amplitude scale to a different scale " \
                           "(Power, Amplitude or dB)."
        self.canRunInBackground = True

    def getParameterInfo(self):
        """Define parameter definitions"""

        # First parameter: input raster
        p_inpath = arcpy.Parameter(
            name="p_inpath",
            displayName="Raster file to be converted",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Input")

        # Second parameter: input scale
        p_inscale = arcpy.Parameter(
            name="p_inscale",
            displayName="Scale of input raster (power or amplitude)",
            datatype="GPString",
            parameterType="Required",
            direction="Input")

        p_inscale.filter.type = "ValueList"
        p_inscale.filter.list = ["Power", "Amplitude"]

        # Third parameter: output scale
        p_outscale = arcpy.Parameter(
            name="p_outscale",
            displayName="Scale of output raster (power, amplitude or dB)",
            datatype="GPString",
            parameterType="Required",
            direction="Input")

        p_outscale.filter.type = "ValueList"
        p_outscale.filter.list = ["Power", "Amplitude", "dB"]

        # Fourth parameter: target directory for output raster
        p_outdir = arcpy.Parameter(
            name="p_outdir",
            displayName="Destination folder for output raster",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input")

        # Fifth parameter: file name for output raster
        p_outname = arcpy.Parameter(
            name="p_outname",
            displayName="File name for output raster (must include valid raster file extension)",
            datatype="GPString",
            parameterType="Required",
            direction="Input")

        # Sixth parameter: select if output is added to the map
        outYN = arcpy.Parameter(
            name="outYN",
            displayName="Add output to map",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input")

        outYN.value = "true"

        # Seventh parameter: output layer to add to project
        outlayer = arcpy.Parameter(
            name="outlayer",
            displayName="Derived output for final product raster",
            datatype="GPRasterLayer",
            parameterType="Derived",
            direction="Output")

        params = [p_inpath, p_inscale, p_outscale, p_outdir, p_outname, outYN, outlayer]
        return params

    def rasConvert(self):
        """Conversion function"""

        if self.inscale == 'Power':
            if self.outscale == 'Amplitude':
                outSqRt = arcpy.sa.SquareRoot(self.inpath)
                outSqRt.save(self.outpath)

            elif self.outscale == 'dB':
                outLog10 = arcpy.sa.Log10(self.inpath)
                outT10 = arcpy.sa.Times(outLog10, 10)
                outT10.save(self.outpath)

        elif self.inscale == 'Amplitude':
            if self.outscale == 'Power':
                outSquare = arcpy.sa.Square(self.inpath)
                outSquare.save(self.outpath)

            elif self.outscale == 'dB':
                outSquare = arcpy.sa.Square(self.inpath)
                outLog10 = arcpy.sa.Log10(outSquare)
                outT10 = arcpy.sa.Times(outLog10, 10)
                outT10.save(self.outpath)

        else:
            arcpy.AddMessage('Parameters entered incorrectly; no conversion performed.')

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

        # Set the default value for p_outdir to be the directory of the input raster
        if parameters[0].value:
            workspace = os.path.dirname(parameters[0].value.value)
            if not parameters[3].altered:
                parameters[3].value = workspace

        # Set the default scale for the input file to be selected based on the p_inpath filename
        if parameters[0].value:
            indirbase = os.path.splitext(os.path.basename(parameters[0].value.value))[0]
            inscale = indirbase[36]
            if inscale == 'a':
                insc = 'Amplitude'
            elif inscale == 'p':
                insc = 'Power'
            else:
                insc = ''
            if not parameters[1].altered:
                parameters[1].value = insc

        # Set the default value for p_outname to be the input raster basename with an output scale tag
        if parameters[2].value:
            scaleTag = parameters[2].value
            if not parameters[4].altered:
                outnm = os.path.splitext(os.path.basename(parameters[0].value.value))[0] + "_" + scaleTag + ".tif"
                parameters[4].value = outnm

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
        self.inpath = parameters[0].valueAsText
        self.inscale = parameters[1].valueAsText
        self.outscale = parameters[2].valueAsText
        self.outdir = parameters[3].valueAsText
        self.outname = parameters[4].valueAsText
        self.outYN = parameters[5].valueAsText

        self.outpath = self.outdir + "\\" + self.outname

        arcpy.AddMessage(
            "Parameters accepted. Converting raster from %s scale to %s scale..." % (self.inscale, self.outscale))

        # Run the rasConvert function to convert from inscale to outscale
        self.rasConvert()

        # Add the output product to the map
        if self.outYN == "true":
            dispname = os.path.splitext(self.outname)[0]
            arcpy.MakeRasterLayer_management(self.outpath, dispname)
            arcpy.SetParameterAsText(6, dispname)
            arcpy.AddMessage("Added converted raster layer to map display.")
        else:
            arcpy.AddMessage(
                "Option to add output layer to map was not selected. "
                "Output can be added manually if desired: %s" % self.outpath)

        # Indicate process is complete
        arcpy.AddMessage("Converted raster from %s scale to %s scale." % (self.inscale, self.outscale))

        # Check In Spatial Analyst Extension
        status = arcpy.CheckInExtension("Spatial")
        arcpy.AddMessage("The Spatial Analyst Extension is in %s status." % status)

        return
