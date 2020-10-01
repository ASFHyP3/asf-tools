import os
import zipfile

import arcpy  # noqa import neccesary but arcpy only available in arcgis environment


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
                messages.addMessage("Extracting files from " + file_name + "...")
                zip_ref.extractall(outlocation)
                zip_ref.close()
                messages.addMessage("Files extracted. Deleting zip folder " + file_name + " from original location...")
                os.remove(file_name)
                messages.addMessage("Unzip complete for " + file_name)

        messages.addMessage("All extractions complete.")

        return
