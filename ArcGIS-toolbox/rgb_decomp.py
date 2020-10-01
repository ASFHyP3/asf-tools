import math
import os
import sys

import arcpy  # noqa import neccesary but arcpy only available in arcgis environment


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
            displayName="Scale of input RTC (Amplitude or Power)",
            datatype="GPString",
            parameterType="Required",
            direction="Input")

        scale.filter.type = "ValueList"
        scale.filter.list = ["Power", "Amplitude"]

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
        outYN = arcpy.Parameter(
            name="outYN",
            displayName="Add output to map",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input")

        outYN.value = "true"

        # Eighth parameter: output layer to add to project
        outlayer = arcpy.Parameter(
            name="outlayer",
            displayName="Derived output for final product raster",
            datatype="GPRasterLayer",
            parameterType="Derived",
            direction="Output")

        params = [indir, scale, pol, rb_thresh_db, outdir, outname, outYN, outlayer]
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

        # Set the default scale for the input file to be selected based on the indir name
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

        # Set the default primary polarization
        if parameters[0].value:
            pol = indirbase[24]
            if not parameters[2].altered:
                parameters[2].value = pol

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
        outYN = parameters[6].valueAsText

        outmsg1 = "Parameters accepted. Generating RGB Decomposition %s..." % outname
        messages.addMessage(outmsg1)

        # Run the code to generate the RGB Decomposition file

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

        elif scale == 'Power':
            cps = cp
            xps = xp
            arcpy.AddMessage("Input scale is power. No conversion necessary.")

        else:
            arcpy.AddError("Input scale must be set to Amplitude or Power.")
            sys.exit(0)

        arcpy.AddMessage("Input rasters have been defined. Running pixel cleanup routine...")

        # Peform pixel cleanup on VV and VH RTC images, using -48 dB as cutoff for valid pixels
        pc_thresh = math.pow(10, -4.8)
        wc_pc = "VALUE < %s" % (pc_thresh)
        # OR: wc_pc = "VALUE < " + str(pc_thresh)
        cp0 = arcpy.sa.Con(cps, 0, cps, wc_pc)
        xp0 = arcpy.sa.Con(xps, 0, xps, wc_pc)

        arcpy.AddMessage("Pixel cleanup complete. Generating spatial masks...")

        # Generate spatial masks based on red/blue threshold
        rb_thresh = math.pow(10, rb_thresh_db / 10)

        # MB = xp0 < k
        remap_mb = "0 %s 1;%s 100000 0" % (rb_thresh, rb_thresh)
        MB = arcpy.sa.Reclassify(xp0, "VALUE", remap_mb, "DATA")

        # MR = xp0 > k
        remap_mr = "0 %s 0;%s 100000 1" % (rb_thresh, rb_thresh)
        MR = arcpy.sa.Reclassify(xp0, "VALUE", remap_mr, "DATA")

        # MX = SXP > 0
        MX = arcpy.sa.Con(xp0, "1", "0", "VALUE > 0")

        arcpy.AddMessage("Spatial masks generated. Deriving red and blue components of surface scatter...")

        # The surface scattering component is divided into red and blue sections
        # Negative values are set to zero
        PR = arcpy.sa.Con((cp0 - (3 * xp0)), "0", (cp0 - (3 * xp0)), "VALUE < 0")
        PB = arcpy.sa.Con(((3 * xp0) - cp0), "0", ((3 * xp0) - cp0), "VALUE < 0")

        # Calculate the difference between the co- and cross-pol values
        # Negative values are set to zero
        sd = arcpy.sa.Con((cp0 - xp0), "0", (cp0 - xp0), "VALUE < 0")

        arcpy.AddMessage(
            "Red and blue components have been derived. Applying spatial masks and scalars for each band...")

        # Apply spatial masks and specific scalars to stretch the values for each band from 1 to 255
        z = 2 / math.pi * MB * arcpy.sa.ATan(arcpy.sa.SquareRoot(sd))
        iR = 254 * MX * (2 * MR * arcpy.sa.SquareRoot(PR) + z) + 1
        iG = 254 * MX * (3 * MR * arcpy.sa.SquareRoot(xp0) + (2 * z)) + 1
        iB = 254 * MX * (2 * arcpy.sa.SquareRoot(PB) + (5 * z)) + 1

        arcpy.AddMessage(
            "Spatial masks and scalars have been applied. Converting bands to 8-bit unsigned integer GeoTIFFs...")

        # Create empty list for RGB bands
        bandList = []

        # Remove negative values and convert each band to an integer raster
        aR = arcpy.sa.Int(arcpy.sa.Con(iR, "255", iR, "VALUE > 255"))
        aG = arcpy.sa.Int(arcpy.sa.Con(iG, "255", iG, "VALUE > 255"))
        aB = arcpy.sa.Int(arcpy.sa.Con(iB, "255", iB, "VALUE > 255"))

        # Save bands as GeoTIFF rasters in the scratch folder
        aRpath = os.path.join(scratchpath, "aR.tif")
        aR.save(aRpath)
        bandList.append(aRpath)

        aGpath = os.path.join(scratchpath, "aG.tif")
        aG.save(aGpath)
        bandList.append(aGpath)

        aBpath = os.path.join(scratchpath, "aB.tif")
        aB.save(aBpath)
        bandList.append(aBpath)

        arcpy.AddMessage(
            "GeoTIFF files for each band have been saved. Combining single-band rasters to generate RGB image...")

        # Combine the aRGB bands into a composite raster
        outpath = os.path.join(outdir, outname)
        arcpy.CompositeBands_management(bandList, outpath)

        # Add the output product to the map
        if outYN == "true":
            dispname = os.path.splitext(outname)[0]
            arcpy.MakeRasterLayer_management(outpath, dispname)
            arcpy.SetParameterAsText(7, dispname)
            arcpy.AddMessage("Added RGB raster layer to map display.")
        else:
            arcpy.AddMessage(
                "Option to add output layer to map was not selected. "
                "Output can be added manually if desired: %s" % outpath)

        arcpy.AddMessage("RGB Decomposition product has been generated: %s. Cleaning up..." % outpath)

        # Delete temporary files
        # The deletion of the temp folder could be set as an option, if there's any chance that users might
        # want to have access to the individual color bands. I don't think that's likely, though.

        # This does not currently work in the Python Toolbox environment. I don't know why
        # If I run the same command in the python window, it behaves as expected. ???
        # The setting of the workspace is probably redundant; it was a hail mary attempt to get it to work.
        arcpy.env.workspace = indir  # delete this if proven to be redundant
        arcpy.Delete_management("temp")

        # Check In Spatial Analyst Extension
        status = arcpy.CheckInExtension("Spatial")
        messages.addMessage("The Spatial Analyst Extension is in %s status." % status)

        # Indicate process is complete
        arcpy.AddMessage("RGB Decomposition process is complete.")

        return
