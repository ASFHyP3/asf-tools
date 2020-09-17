#################################
##  ASF ArcGIS Toolbox         ##
##  Heidi Kristenson           ##
##  Alaska Satellite Facility  ##
##  16 September 2020          ##
#################################

import arcpy, os, zipfile

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
        self.description = "This tool unzips ASF products downloaded from Vertex or HyP3 and saves the extracted folders to the desired destination folder."
        self.canRunInBackground = True

    def getParameterInfo(self):
        """Define parameter definitions"""

        # First parameter: location of downloaded files
        ziplocation = arcpy.Parameter(
            name = "ziplocation",
            displayName = "Location of .zip files",
            datatype = "DEWorkspace",
            parameterType = "Required",
            direction = "Input")

        ziplocation.filter.list = ["File System"]
        
        # If you would like to set the value to your default download folder, enter the path between the single quotes below.
        # Example: ziplocation.value = r'C:\Users\YourUserID\Downloads'
        ziplocation.value = r''

        # Second parameter: destination folder for extracted products
        outlocation = arcpy.Parameter(
            name = "outlocation",
            displayName = "Destination folder for extracted products",
            datatype = "DEWorkspace",
            parameterType = "Required",
            direction = "Input")

        outlocation.filter.list = ["File System"]
        
        # If you would like to set a default value for your destination folder, enter the path between the single quotes below.
        # Example: outlocation.value = r'F:\Projects\VertexData'
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
                messages.addMessage("Extracting files from "+file_name+"...")
                zip_ref.extractall(outlocation)
                zip_ref.close()
                messages.addMessage("Files extracted. Deleting zip folder "+file_name+" from original location...")
                os.remove(file_name)
                messages.addMessage("Unzip complete for "+file_name)

        messages.addMessage("All extractions complete.")
        
        return

class ScaleConversion(object):
    def __init__(self):
        
        """Converts RTC products from power to amplitude"""
        self.label = "Scale Conversion (Power, Amplitude, dB)"
        self.description = "This tool converts RTC products from Power or Amplitude scale to a different scale (Power, Amplitude or dB)."
        self.canRunInBackground = True       

    def getParameterInfo(self):
        """Define parameter definitions"""

        # First parameter: input raster
        p_inpath = arcpy.Parameter(
            name = "p_inpath",
            displayName = "Raster file to be converted",
            datatype = "DERasterDataset",
            parameterType = "Required",
            direction = "Input")

        # Second parameter: input scale
        p_inscale = arcpy.Parameter(
            name = "p_inscale",
            displayName = "Scale of input raster (power or amplitude)",
            datatype = "GPString",
            parameterType = "Required",
            direction = "Input")
        
        p_inscale.filter.type = "ValueList"
        p_inscale.filter.list = ["Power", "Amplitude"]

        # Third parameter: output scale
        p_outscale = arcpy.Parameter(
            name = "p_outscale",
            displayName = "Scale of output raster (power, amplitude or dB)",
            datatype = "GPString",
            parameterType = "Required",
            direction = "Input")
        
        p_outscale.filter.type = "ValueList"
        p_outscale.filter.list = ["Power", "Amplitude", "dB"]

        # Fourth parameter: target directory for output raster
        p_outdir = arcpy.Parameter(
            name = "p_outdir",
            displayName = "Destination folder for output raster",
            datatype = "DEWorkspace",
            parameterType = "Required",
            direction = "Input")

        # Fifth parameter: file name for output raster
        p_outname  = arcpy.Parameter(
            name = "p_outname",
            displayName = "File name for output raster (must include valid raster file extension)",
            datatype = "GPString",
            parameterType = "Required",
            direction = "Input")
                        
        params = [p_inpath, p_inscale, p_outscale, p_outdir, p_outname]
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
            arcpy.AddMessage("Spatial Analyst extension is not available for use. Check your licensing to make sure you have access to this extension.")
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

        # Set the default scale for the input file to be selected based on the indir name
        # THIS NEEDS TO BE MADE MORE ROBUST; IF CHARACTER 36 ISN'T A OR P, IT THROWS AN ERROR
        # AND MAKES EVERYTHING SAD. MAYBE A TRY CLAUSE? OR EXCEPTION?
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
                outnm = os.path.splitext(os.path.basename(parameters[0].value.value))[0]+"_"+scaleTag+".tif"
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

        self.outpath = self.outdir+"\\"+self.outname

        txt_msg1 = "Parameters accepted. Converting raster from %s scale to %s scale..." % (self.inscale, self.outscale)
        messages.addMessage(txt_msg1)
        
        # Run the rasConvert function to convert from inscale to outscale
        self.rasConvert()
        
        # Indicate process is complete
        txt_msg2 = "Converted raster from %s scale to %s scale." % (self.inscale, self.outscale)
        messages.addMessage(txt_msg2)

        # Check In Spatial Analyst Extension
        status = arcpy.CheckInExtension("Spatial")
        arcpy.AddMessage("The Spatial Analyst Extension is in %s status." %status)

        return

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
            name = "inRTC",
            displayName = "Raster to be reclassifed",
            datatype = "DERasterDataset",
            parameterType = "Required",
            direction = "Input")

        # Second parameter: output path for reclassified raster
        rc_outpath = arcpy.Parameter(
            name = "rc_outpath",
            displayName = "Destination folder for reclassified raster",
            datatype = "DEWorkspace",
            parameterType = "Required",
            direction = "Input")

        # Third parameter: output file name for reclassified raster
        rc_outname  = arcpy.Parameter(
            name = "rc_outname",
            displayName = "File name for reclassified raster (must include valid raster file extension)",
            datatype = "GPString",
            parameterType = "Required",
            direction = "Input")
	
        # Fourth parameter: threshold value
        thresh = arcpy.Parameter(
            name = "thresh",
            displayName = "Threshold value for reclassification",
            datatype = "GPDouble",
            parameterType = "Required",
            direction = "Input")
                        
        params = [inRTC, rc_outpath, rc_outname, thresh]
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
            arcpy.AddMessage("Spatial Analyst extension is not available for use. Check your licensing to make sure you have access to this extension.")
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
                outnmR = os.path.splitext(os.path.basename(parameters[0].value.value))[0]+"_Reclass.tif"
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
        thresh = parameters[3].value
        threshstr = str(thresh)

        # Run the code to reclassify the image
        
        rcname = str(rc_outpath+'\\'+rc_outname)
        values = "-1000.000000 %s 1;%s 1000.000000 NODATA" % (threshstr, threshstr)
        reclass = arcpy.gp.Reclassify_sa(inRTC, "VALUE", values, rcname, "DATA")

        # Indicate process is complete
        txt_msg3 = "Reclassified raster generated for %s." % (inRTC)
        messages.addMessage(txt_msg3)

        # Check In Spatial Analyst Extension
        status = arcpy.CheckInExtension("Spatial")
        arcpy.AddMessage("The Spatial Analyst Extension is in %s status." %status)
        
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
            name = "date2",
            displayName = "Comparison raster (i.e. most recent SAR acquisition)",
            datatype = "DERasterDataset",
            parameterType = "Required",
            direction = "Input")

        # Second parameter: oldest RTC file
        date1 = arcpy.Parameter(
            name = "date1",
            displayName = "Reference raster (i.e. oldest SAR acquisition)",
            datatype = "DERasterDataset",
            parameterType = "Required",
            direction = "Input")

        # Third parameter: output path for log difference file
        outdir = arcpy.Parameter(
            name = "outdir",
            displayName = "Destination folder for output file",
            datatype = "DEWorkspace",
            parameterType = "Required",
            direction = "Input")

        # Fourth parameter: output file name for log difference file
        outname = arcpy.Parameter(
            name = "outname",
            displayName = "File name for output log difference file (including valid raster file extension)",
            datatype = "GPString",
            parameterType = "Required",
            direction = "Input")
                        
        params = [date2, date1, outdir, outname]
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
            arcpy.AddMessage("Spatial Analyst extension is not available for use. Check your licensing to make sure you have access to this extension.")
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
            outflnm = str(d1base+'_'+d2base+'_LogDiff.tif')
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

        outmsg1 = "Parameters accepted. Generating Log Difference file %s..." %outname
        messages.addMessage(outmsg1)
	
        # Run the code to calculate the log difference
        outLogDiff = str(outdir+'\\'+outname)
        outLog10 = arcpy.sa.Log10(arcpy.sa.Divide(date2, date1))
        outLog10.save(outLogDiff)

        # Indicate process is complete
        outmsg2 = "Log Difference raster %s generated." %outname

        # Check In Spatial Analyst Extension
        status = arcpy.CheckInExtension("Spatial")
        messages.addMessage("The Spatial Analyst Extension is in %s status." %status)

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
            name = "indir",
            displayName = "Input directory containing dual-pol RTC data",
            datatype = "DEWorkspace",
            parameterType = "Required",
            direction = "Input")

        # Second parameter: scale of input dataset
        scale = arcpy.Parameter(
            name = "scale",
            displayName = "Scale of input RTC (amplitude or power)",
            datatype = "GPString",
            parameterType = "Required",
            direction = "Input")

        # Third parameter: R/B threshold in dB
        rb_thresh_db = arcpy.Parameter(
            name = "rb_thresh_db",
            displayName = "Threshold cutoff value for red/blue in dB (default: -24 dB)",
            datatype = "GPLong",
            parameterType = "Required",
            direction = "Input")

        # Fourth parameter: output directory for RGB file
        outdir = arcpy.Parameter(
            name = "outdir",
            displayName = "Output directory for new RGB file",
            datatype = "DEWorkspace",
            parameterType = "Required",
            direction = "Input")

        # Fifth parameter: output name for RGB file
        outname = arcpy.Parameter(
            name = "outname",
            displayName = "Filename for new RGB raster",
            datatype = "GPString",
            parameterType = "Required",
            direction = "Input")

        params = [indir, scale, rb_thresh_db, outdir, outname]
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
            arcpy.AddMessage("Spatial Analyst extension is not available for use. Check your licensing to make sure you have access to this extension.")
            return False

        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""

        # Set the default scale for the input file to be selected based on the indir name
        # THIS NEEDS TO BE MADE MORE ROBUST; IF CHARACTER 36 ISN'T A OR P, IT THROWS AN ERROR
        # AND MAKES EVERYTHING SAD. MAYBE A TRY CLAUSE? OR EXCEPTION?
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

        # Set the default R/B threshold value
        if not parameters[2].altered:
            parameters[2].value = -24

        # Set the default output directory to be the same as the input directory
        if parameters[0].value:
            if not parameters[3].altered:
                parameters[3].value = parameters[0].value.value

        # Set the default output filename to be the basename_RGB.tif
        if parameters[0].value:
            if not parameters[4].altered:
                parameters[4].value = ("%s_RGB.tif" % indirbase)

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
        rb_thresh_db = parameters[2].valueAsText
        outdir = parameters[3].valueAsText
        outname = parameters[4].valueAsText

        outmsg1 = "Parameters accepted. Generating RGB Decomposition %s..." %outname
        messages.addMessage(outmsg1)

        # Run the code to generate the RGB Decomposition file

        # Create a message generation function
        def logMessage(msg):
            print(msg)
            arcpy.AddMessage(msg)
            messages.addMessage(msg)

        logMessage("Input parameters have been defined. Preparing workspace...")

        # Set the working directory containing the RTC images
        arcpy.env.workspace = indir
        # Create a scratch directory for intermediate files
        scratch = arcpy.CreateFolder_management(indir, "temp")
        scratchpath = os.path.join(indir, "temp")

        logMessage("Workspace has been prepared. Defining input rasters...")

        # Set variables for co-pol and cross-pol GeoTIFFs
        ## Figure out how to elegantly determine if the dataset is VV/VH or HH/HV
        ### Could be either a user-set parameter, but would be better to determine from the directory contents
        ## For now, just assuming VV/VH

        vv_tif = arcpy.ListRasters('*VV.tif')[0]
        #hh_tif = arcpy.ListRasters('*HH.tif')[0]
        cp = os.path.join(indir, vv_tif)

        vh_tif = arcpy.ListRasters('*VH.tif')[0]
        #hv_tif = arcpy.ListRasters('*HV.tif')[0]
        xp = os.path.join(indir, vh_tif)

        logMessage("Input rasters have been defined. Running pixel cleanup routine...")

        # Peform pixel cleanup on VV and VH RTC images, using -48 dB as cutoff for valid pixels
        pc_thresh = math.pow(10, -4.8)
        wc_pc = "VALUE < " + str(pc_thresh)
        #OR: wc_pc = "VALUE < %s" % (pc_thresh)
        cp0 = arcpy.sa.Con(cp, 0, cp, wc_pc)
        xp0 = arcpy.sa.Con(xp, 0, xp, wc_pc)

        logMessage("Pixel cleanup complete. Generating spatial masks...")

        # Generate spatial masks based on red/blue threshold
        rb_thresh = math.pow(10, -2.4)

        # MB = xp0 < k
        remap_mb = "0 %s 1;%s 100000 0" %(rb_thresh, rb_thresh)
        MB = arcpy.sa.Reclassify(xp0, "VALUE", remap_mb, "DATA")

        # MR = xp0 > k
        remap_mr = "0 %s 0;%s 100000 1" %(rb_thresh, rb_thresh)
        MR = arcpy.sa.Reclassify(xp0, "VALUE", remap_mr, "DATA")

        # MX = SXP > 0
        MX = arcpy.sa.Con(xp0, "1", "0", "VALUE > 0")

        logMessage("Spatial masks generated. Deriving red and blue components of surface scatter...")

        # The surface scattering component is divided into red and blue sections
        # Negative values are set to zero
        PR = arcpy.sa.Con((cp0 - (3*xp0)), "0", (cp0 - (3*xp0)), "VALUE < 0")
        PB = arcpy.sa.Con(((3*xp0) - cp0), "0", ((3*xp0) - cp0), "VALUE < 0")

        # Calculate the difference between the co- and cross-pol values
        # Negative values are set to zero
        sd = arcpy.sa.Con((cp0 - xp0), "0", (cp0 - xp0), "VALUE < 0")

        logMessage("Red and blue components have been derived. Applying spatial masks and scalars for each band...")

        # Apply spatial masks and specific scalars to stretch the values for each band from 1 to 255
        z = 2 / math.pi * MB * arcpy.sa.ATan(arcpy.sa.SquareRoot(sd))
        iR = 254 * MX * (2 * MR * arcpy.sa.SquareRoot(PR) + z) + 1
        iG = 254 * MX * (3 * MR * arcpy.sa.SquareRoot(xp0) + (2 * z)) + 1
        iB = 254 * MX * (2 * arcpy.sa.SquareRoot(PB) + (5 * z)) + 1

        logMessage("Spatial masks and scalars have been applied. Converting bands to 8-bit unsigned integer GeoTIFFs...")

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

        logMessage("GeoTIFF files for each band have been saved. Combining single-band rasters to generate RGB image...")

        # Combine the aRGB bands into a composite raster
        outpath = os.path.join(outdir, outname)
        arcpy.CompositeBands_management(bandList, outpath)

        logMessage("RGB Decomposition product has been generated: %s" % outpath)

        # Indicate process is complete
        outmsg2 = "Log Difference raster %s generated." %outname

        # Check In Spatial Analyst Extension
        status = arcpy.CheckInExtension("Spatial")
        messages.addMessage("The Spatial Analyst Extension is in %s status." %status)

        return