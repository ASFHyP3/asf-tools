############################################
#   ASF ArcGIS Toolbox                     #
#   Heidi Kristenson, ASF Tools Team       #
#   uaf-asf-apd@alaska.edu                 #
#   Alaska Satellite Facility              #
#   https://github.com/ASFHyP3/asf-tools   #
############################################
import math
import os
import sys
import zipfile

import arcpy  # noqa import necessary but arcpy only available in arcgis environment


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the .pyt file)."""
        self.label = "ASF Tools"
        self.alias = "ASF Tools"

        # List of tool classes associated with this toolbox
        self.tools = [UnzipFiles, ScaleConversion, ReclassifyRTC, LogDiff, RGBDecomp]


class UnzipFiles(object):
    def __init__(self):

        """Unzips ASF products downloaded from Vertex or HyP3."""
        self.label = "Unzip Files"
        self.description = "This tool unzips ASF products downloaded from Vertex or HyP3 and saves the extracted " \
                           "folders to the desired destination folder. "
        self.canRunInBackground = True

    def getParameterInfo(self):
        """Define parameter definitions"""

        # First parameter: location of downloaded files
        ziplocation = arcpy.Parameter(
            name="ziplocation",
            displayName="Location of .zip files",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input")

        ziplocation.filter.list = ["File System"]

        # If you would like to set the value to your default download folder, enter the path between the single
        # quotes below. Example: ziplocation.value = r'C:\Users\YourUserID\Downloads'
        ziplocation.value = r''

        # Second parameter: destination folder for extracted products
        outlocation = arcpy.Parameter(
            name="outlocation",
            displayName="Destination folder for extracted products",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input")

        outlocation.filter.list = ["File System"]

        # If you would like to set a default value for your destination folder, enter the path between the single
        # quotes below. Example: outlocation.value = r'F:\Projects\VertexData'
        outlocation.value = r''

        params = [ziplocation, outlocation]
        return params

    def isLicensed(self):
        """This tool does not require any special licenses or extensions."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""

        # Define parameters
        ziplocation = parameters[0].valueAsText
        outlocation = parameters[1].valueAsText

        # Run the code to unzip the .zip products in the download folder to the desired destination
        os.chdir(ziplocation)
        for item in os.listdir(ziplocation):
            if item.endswith('.zip'):
                file_name = os.path.abspath(item)
                zip_ref = zipfile.ZipFile(file_name)
                arcpy.AddMessage("Extracting files from " + file_name + "...")
                zip_ref.extractall(outlocation)
                zip_ref.close()
                arcpy.AddMessage("Files extracted. Deleting zip folder " + file_name + " from original location...")
                os.remove(file_name)
                arcpy.AddMessage("Unzip complete for " + file_name)

        arcpy.AddMessage("All extractions complete.")

        return


class ScaleConversion(object):
    def __init__(self):

        """Converts RTC products between Power, Amplitude, and dB scales"""
        self.label = "Scale Conversion (Power, Amplitude, dB)"
        self.description = "This tool converts RTC products from Power, Amplitude, or dB scale to a different scale " \
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
            displayName="Scale of input raster (power, amplitude, or dB)",
            datatype="GPString",
            parameterType="Required",
            direction="Input")

        p_inscale.filter.type = "ValueList"
        p_inscale.filter.list = ["Power", "Amplitude", "dB"]

        # Third parameter: output scale
        p_outscale = arcpy.Parameter(
            name="p_outscale",
            displayName="Scale of output raster (power, amplitude, or dB)",
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
        out_yn = arcpy.Parameter(
            name="out_yn",
            displayName="Add output to map",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input")

        out_yn.value = "true"

        # Seventh parameter: output layer to add to project
        outlayer = arcpy.Parameter(
            name="outlayer",
            displayName="Derived output for final product raster",
            datatype="GPRasterLayer",
            parameterType="Derived",
            direction="Output")

        params = [p_inpath, p_inscale, p_outscale, p_outdir, p_outname, out_yn, outlayer]
        return params

    def rasConvert(self):
        """Conversion function"""

        if self.inscale == 'Power':
            if self.outscale == 'Amplitude':
                out_sqrt = arcpy.sa.SquareRoot(self.inpath)
                out_sqrt.save(self.outpath)

            elif self.outscale == 'dB':
                out_log10 = arcpy.sa.Log10(self.inpath)
                out_t10 = arcpy.sa.Times(out_log10, 10)
                out_t10.save(self.outpath)

        elif self.inscale == 'Amplitude':
            if self.outscale == 'Power':
                out_square = arcpy.sa.Square(self.inpath)
                out_square.save(self.outpath)

            elif self.outscale == 'dB':
                out_square = arcpy.sa.Square(self.inpath)
                out_log10 = arcpy.sa.Log10(out_square)
                out_t10 = arcpy.sa.Times(out_log10, 10)
                out_t10.save(self.outpath)

        elif self.inscale == 'dB':
            if self.outscale == 'Power':
                out_d10 = arcpy.sa.Divide(self.inpath, 10)
                out_exp = arcpy.sa.Power(10, out_d10)
                out_exp.save(self.outpath)

            elif self.outscale == 'Amplitude':
                out_d20 = arcpy.sa.Divide(self.inpath, 20)
                out_exp = arcpy.sa.Power(10, out_d20)
                out_exp.save(self.outpath)

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
        except Exception:  # noqa: B902
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
            inrasbase = os.path.splitext(os.path.basename(parameters[0].value.value))[0]
            inscale = inrasbase[36]
            if inscale == 'a':
                insc = 'Amplitude'
            elif inscale == 'p':
                insc = 'Power'
            elif inscale == 'd':
                insc = 'dB'
            else:
                insc = ''
            if not parameters[1].altered:
                parameters[1].value = insc

        # Set the default value for p_outname to be the input raster basename with an output scale tag
        if parameters[2].value:
            scale_tag = parameters[2].value
            if not parameters[4].altered:
                outnm = os.path.splitext(os.path.basename(parameters[0].value.value))[0] + "_" + scale_tag + ".tif"
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
        self.out_yn = parameters[5].valueAsText

        self.outpath = self.outdir + "\\" + self.outname

        arcpy.AddMessage(
            "Parameters accepted. Converting raster from %s scale to %s scale..." % (self.inscale, self.outscale))

        # Run the rasConvert function to convert from inscale to outscale
        self.rasConvert()

        # Indicate process is complete
        arcpy.AddMessage("Converted raster from %s scale to %s scale." % (self.inscale, self.outscale))

        # Add the output product to the map
        if self.out_yn == "true":
            dispname = os.path.splitext(self.outname)[0]
            arcpy.MakeRasterLayer_management(self.outpath, dispname)
            arcpy.SetParameterAsText(6, dispname)
            arcpy.AddMessage("Added converted raster layer to map display.")
        else:
            arcpy.AddMessage(
                "Option to add output layer to map was not selected. "
                "Output can be added manually if desired: %s" % self.outpath)

        # Check In Spatial Analyst Extension
        status = arcpy.CheckInExtension("Spatial")
        arcpy.AddMessage("The Spatial Analyst Extension is in %s status." % status)

        return


class ReclassifyRTC(object):
    def __init__(self):

        """Reclassifies Raster to apply pixel value of 1 to pixels with original values below a threshold"""
        self.label = "Reclassify RTC"
        self.description = "This tool generates a reclassified raster based on a threshold value."
        self.canRunInBackground = True

    def getParameterInfo(self):
        """Define parameter definitions"""

        # First parameter: input RTC file to be reclassified
        in_rtc = arcpy.Parameter(
            name="in_rtc",
            displayName="Raster to be reclassified",
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
        out_yn = arcpy.Parameter(
            name="out_yn",
            displayName="Add output to map",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input")

        out_yn.value = "true"

        # Sixth parameter: output layer to add to project
        outlayer = arcpy.Parameter(
            name="outlayer",
            displayName="Derived output for final product raster",
            datatype="GPRasterLayer",
            parameterType="Derived",
            direction="Output")

        params = [in_rtc, rc_outpath, rc_outname, thresh, out_yn, outlayer]
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
        except Exception:  # noqa: B902
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
            workspace = os.path.dirname(parameters[0].value.value)
            if not parameters[1].altered:
                parameters[1].value = workspace

        # Set the default value for rc_outname to be the basename of the input raster with a Reclass tag
        if parameters[0].value:
            if not parameters[2].altered:
                outnm = os.path.splitext(os.path.basename(parameters[0].value.value))[0] + "_Reclass.tif"
                parameters[2].value = outnm

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
        in_rtc = parameters[0].valueAsText
        rc_outpath = parameters[1].valueAsText
        rc_outname = parameters[2].valueAsText
        thresh = parameters[3].valueAsText
        out_yn = parameters[4].valueAsText

        # Run the code to reclassify the image
        arcpy.AddMessage("Reclassifying raster based on a threshold of %s..." % thresh)
        rcname = os.path.join(rc_outpath, rc_outname)
        values = "-1000.000000 %s 1;%s 1000.000000 NODATA" % (thresh, thresh)
        arcpy.gp.Reclassify_sa(in_rtc, "VALUE", values, rcname, "DATA")

        # Indicate process is complete
        arcpy.AddMessage("Reclassified raster generated for %s." % in_rtc)

        # Add the output product to the map
        if out_yn == "true":
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
        out_yn = arcpy.Parameter(
            name="out_yn",
            displayName="Add output to map",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input")

        out_yn.value = "true"

        # Sixth parameter: output layer to add to project
        outlayer = arcpy.Parameter(
            name="outlayer",
            displayName="Derived output for final product raster",
            datatype="GPRasterLayer",
            parameterType="Derived",
            direction="Output")

        params = [date2, date1, outdir, outname, out_yn, outlayer]
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
        except Exception:  # noqa: B902
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
        out_yn = parameters[4].valueAsText

        arcpy.AddMessage("Parameters accepted. Generating Log Difference file %s..." % outname)

        # Run the code to calculate the log difference
        out_logdiff = os.path.join(outdir, outname)
        out_log10 = arcpy.sa.Log10(arcpy.sa.Divide(date2, date1))
        out_log10.save(out_logdiff)

        # Indicate process is complete
        arcpy.AddMessage("Log Difference raster %s generated." % outname)

        # Add the output product to the map
        if out_yn == "true":
            dispname = os.path.splitext(outname)[0]
            arcpy.MakeRasterLayer_management(out_logdiff, dispname)
            arcpy.SetParameterAsText(5, dispname)
            arcpy.AddMessage("Added Log Difference raster layer to map display.")
        else:
            arcpy.AddMessage(
                "Option to add output layer to map was not selected. "
                "Output can be added manually if desired: %s" % out_logdiff)

        # Check In Spatial Analyst Extension
        status = arcpy.CheckInExtension("Spatial")
        arcpy.AddMessage("The Spatial Analyst Extension is in %s status." % status)

        return


class RGBDecomp(object):
    def __init__(self):

        """Generates an RGB image from co- and cross-pol RTC data"""
        self.label = "RGB Decomposition"
        self.description = "This tool generates an RGB image using co- and cross-pol RTC data."
        self.canRunInBackground = True

    def getParameterInfo(self):
        """Define parameter definitions"""

        # First parameter: input directory
        indir = arcpy.Parameter(
            name="indir",
            displayName="Input directory containing dual-pol RTC data",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input")

        # Second parameter: scale of input dataset
        scale = arcpy.Parameter(
            name="scale",
            displayName="Scale of input RTC (Power, Amplitude, or dB)",
            datatype="GPString",
            parameterType="Required",
            direction="Input")

        scale.filter.type = "ValueList"
        scale.filter.list = ["Power", "Amplitude", "dB"]

        # Third parameter: Primary polarization
        pol = arcpy.Parameter(
            name="pol",
            displayName="Primary polarization (V or H)",
            datatype="GPString",
            parameterType="Required",
            direction="Input")

        pol.filter.type = "ValueList"
        pol.filter.list = ["V", "H"]

        # Fourth parameter: R/B threshold in dB
        rb_thresh_db = arcpy.Parameter(
            name="rb_thresh_db",
            displayName="Threshold cutoff value for red/blue in dB (default: -24.0 dB)",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input")

        # Fifth parameter: output directory for RGB file
        outdir = arcpy.Parameter(
            name="outdir",
            displayName="Output directory for new RGB file",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input")

        # Sixth parameter: output name for RGB file
        outname = arcpy.Parameter(
            name="outname",
            displayName="Filename for new RGB raster",
            datatype="GPString",
            parameterType="Required",
            direction="Input")

        # Seventh parameter: select if output is added to the map
        out_yn = arcpy.Parameter(
            name="out_yn",
            displayName="Add output to map",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input")

        out_yn.value = "true"

        # Eighth parameter: output layer to add to project
        outlayer = arcpy.Parameter(
            name="outlayer",
            displayName="Derived output for final product raster",
            datatype="GPRasterLayer",
            parameterType="Derived",
            direction="Output")

        params = [indir, scale, pol, rb_thresh_db, outdir, outname, out_yn, outlayer]
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
        except Exception:  # noqa: B902
            arcpy.AddMessage(
                "Spatial Analyst extension is not available for use. Check your licensing to make sure you have "
                "access to this extension.")
            return False

        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""

        # Set the default scale for the input file to be selected based on the indir name
        if parameters[0].value:
            indirbase = os.path.basename(parameters[0].value.value)
            inscale = indirbase[36]
            if inscale == 'a':
                insc = 'Amplitude'
            elif inscale == 'p':
                insc = 'Power'
            elif inscale == 'd':
                insc = 'dB'
            else:
                insc = ''
            if not parameters[1].altered:
                parameters[1].value = insc

        # Set the default primary polarization
        if parameters[0].value:
            pol = indirbase[24]
            if pol in ("V", "H"):
                if not parameters[2].altered:
                    parameters[2].value = pol
            else:
                if not parameters[2].altered:
                    parameters[2].value = ''

        # Set the default R/B threshold value
        if not parameters[3].altered:
            parameters[3].value = -24

        # Set the default output directory to be the same as the input directory
        if parameters[0].value:
            if not parameters[4].altered:
                parameters[4].value = parameters[0].value.value

        # Set the default output filename to be the basename_RGB.tif
        if parameters[0].value:
            if not parameters[5].altered:
                parameters[5].value = ("%s_RGB.tif" % indirbase)

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
        indir = parameters[0].valueAsText
        scale = parameters[1].valueAsText
        pol = parameters[2].valueAsText
        rb_thresh_db = parameters[3].value
        outdir = parameters[4].valueAsText
        outname = parameters[5].valueAsText
        out_yn = parameters[6].valueAsText

        arcpy.AddMessage("Input parameters have been defined. Preparing workspace...")

        # Set the working directory containing the RTC images
        arcpy.env.workspace = indir
        indirbase = os.path.basename(indir)
        # Create a scratch directory for intermediate files
        arcpy.CreateFolder_management(indir, "temp")
        scratchpath = os.path.join(indir, "temp")

        arcpy.AddMessage("Workspace has been prepared. Defining input rasters...")

        # Confirm that the input dataset is dual-pol
        dpol = indirbase[23]
        if dpol == 'D':
            arcpy.AddMessage("Based on the input directory, this is a dual-pol dataset.")
        elif dpol == 'S':
            arcpy.AddError("This is a single-pol dataset. A dual-pol dataset is required for RGB Decomposition.")
            sys.exit(0)
        else:
            arcpy.AddMessage("Dual-polarization cannot be determined from the input directory.")

        # Set variables for co-pol and cross-pol GeoTIFFs
        if pol == 'V':
            vv_tif = arcpy.ListRasters('*VV.tif')[0]
            cp = os.path.join(indir, vv_tif)
            arcpy.AddMessage("Co-pol dataset: %s" % vv_tif)
            vh_tif = arcpy.ListRasters('*VH.tif')[0]
            xp = os.path.join(indir, vh_tif)
            arcpy.AddMessage("Cross-pol dataset: %s" % vh_tif)

        elif pol == 'H':
            hh_tif = arcpy.ListRasters('*HH.tif')[0]
            cp = os.path.join(indir, hh_tif)
            arcpy.AddMessage("Co-pol dataset: %s" % hh_tif)
            hv_tif = arcpy.ListRasters('*HV.tif')[0]
            xp = os.path.join(indir, hv_tif)
            arcpy.AddMessage("Cross-pol dataset: %s" % hv_tif)

        # Convert the scale to power if necessary
        if scale == 'Amplitude':
            cps = arcpy.sa.Square(cp)
            xps = arcpy.sa.Square(xp)
            arcpy.AddMessage("Input scale is amplitude. Converted to power scale.")

        elif scale == 'dB':
            cps = arcpy.sa.Power(10, arcpy.sa.Divide(cp, 10))
            xps = arcpy.sa.Power(10, arcpy.sa.Divide(xp, 10))
            arcpy.AddMessage("Input scale is dB. Converted to power scale.")

        elif scale == 'Power':
            cps = cp
            xps = xp
            arcpy.AddMessage("Input scale is power. No conversion necessary.")

        else:
            arcpy.AddError("Input scale must be set to Power, Amplitude, or dB.")
            sys.exit(0)

        arcpy.AddMessage("Input rasters have been defined. Running pixel cleanup routine...")

        # Perform pixel cleanup on VV and VH RTC images, using -48 dB as cutoff for valid pixels
        pc_thresh = math.pow(10, -4.8)
        wc_pc = "VALUE < %s" % (pc_thresh)
        cp0 = arcpy.sa.Con(cps, 0, cps, wc_pc)
        xp0 = arcpy.sa.Con(xps, 0, xps, wc_pc)

        arcpy.AddMessage("Pixel cleanup complete. Generating spatial masks...")

        # Generate spatial masks based on red/blue threshold
        rb_thresh = math.pow(10, rb_thresh_db / 10)

        # mb = xp0 < k
        remap_mb = "0 %s 1;%s 100000 0" % (rb_thresh, rb_thresh)
        mb = arcpy.sa.Reclassify(xp0, "VALUE", remap_mb, "DATA")

        # mr = xp0 > k
        remap_mr = "0 %s 0;%s 100000 1" % (rb_thresh, rb_thresh)
        mr = arcpy.sa.Reclassify(xp0, "VALUE", remap_mr, "DATA")

        # mx = SXP > 0
        mx = arcpy.sa.Con(xp0, "1", "0", "VALUE > 0")

        arcpy.AddMessage("Spatial masks generated. Deriving red and blue components of surface scatter...")

        # The surface scattering component is divided into red and blue sections
        # Negative values are set to zero
        pr = arcpy.sa.Con((cp0 - (3 * xp0)), "0", (cp0 - (3 * xp0)), "VALUE < 0")
        pb = arcpy.sa.Con(((3 * xp0) - cp0), "0", ((3 * xp0) - cp0), "VALUE < 0")

        # Calculate the difference between the co- and cross-pol values
        # Negative values are set to zero
        sd = arcpy.sa.Con((cp0 - xp0), "0", (cp0 - xp0), "VALUE < 0")

        arcpy.AddMessage(
            "Red and blue components have been derived. Applying spatial masks and scalars for each band...")

        # Apply spatial masks and specific scalars to stretch the values for each band from 1 to 255
        z = 2 / math.pi * mb * arcpy.sa.ATan(arcpy.sa.SquareRoot(sd))
        ir = 254 * mx * (2 * mr * arcpy.sa.SquareRoot(pr) + z) + 1
        ig = 254 * mx * (3 * mr * arcpy.sa.SquareRoot(xp0) + (2 * z)) + 1
        ib = 254 * mx * (2 * arcpy.sa.SquareRoot(pb) + (5 * z)) + 1

        arcpy.AddMessage(
            "Spatial masks and scalars have been applied. Converting bands to 8-bit unsigned integer GeoTIFFs...")

        # Create empty list for RGB bands
        band_list = []

        # Remove negative values and convert each band to an integer raster
        ar = arcpy.sa.Int(arcpy.sa.Con(ir, "255", ir, "VALUE > 255"))
        ag = arcpy.sa.Int(arcpy.sa.Con(ig, "255", ig, "VALUE > 255"))
        ab = arcpy.sa.Int(arcpy.sa.Con(ib, "255", ib, "VALUE > 255"))

        # Save bands as GeoTIFF rasters in the scratch folder
        arpath = os.path.join(scratchpath, "ar.tif")
        ar.save(arpath)
        band_list.append(arpath)

        agpath = os.path.join(scratchpath, "ag.tif")
        ag.save(agpath)
        band_list.append(agpath)

        abpath = os.path.join(scratchpath, "ab.tif")
        ab.save(abpath)
        band_list.append(abpath)

        arcpy.AddMessage(
            "GeoTIFF files for each band have been saved. Combining single-band rasters to generate RGB image...")

        # Combine the aRGB bands into a composite raster
        outpath = os.path.join(outdir, outname)
        arcpy.CompositeBands_management(band_list, outpath)
        arcpy.AddMessage("RGB Decomposition product has been generated: %s." % outpath)

        # Indicate process is complete
        arcpy.AddMessage("RGB Decomposition process is complete.")

        # Add the output product to the map
        if out_yn == "true":
            dispname = os.path.splitext(outname)[0]
            arcpy.MakeRasterLayer_management(outpath, dispname)
            arcpy.SetParameterAsText(7, dispname)
            arcpy.AddMessage("Added RGB raster layer to map display.")
        else:
            arcpy.AddMessage(
                "Option to add output layer to map was not selected. "
                "Output can be added manually if desired: %s" % outpath)

        """
        # Delete temporary files
        arcpy.AddMessage("Cleaning up...")

        # The deletion of the temp folder could be set as an option, if there's any chance that users might
        # want to have access to the individual color bands. I don't think that's likely, though.

        # There is not currently a mechanism for getting rid of the temp folder.
        # The arcpy.DeleteManagement approach doesn't work from within a tool, so that's not an option.
        # There seems to be a residual hold on at least one of the saved input bands for the RGB Composite,
        # so using shutil doesn't work (it won't delete a directory if a file contained in the directory
        # still has a file lock applied). The lock is probably ArcGIS-driven.
        # Not sure how best to proceed to get rid of the temp folder once the process is complete.

        arcpy.AddMessage("Temporary files have been deleted.")
        """

        # Check In Spatial Analyst Extension
        status = arcpy.CheckInExtension("Spatial")
        arcpy.AddMessage("The Spatial Analyst Extension is in %s status." % status)

        return
