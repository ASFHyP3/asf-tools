ArcGIS Toolbox
==============

[![DOI](https://zenodo.org/badge/295506894.svg)](https://zenodo.org/badge/latestdoi/295506894)

The [ASF_Tools ArcGIS Python Toolbox](https://asf.alaska.edu/how-to/data-tools/gis-tools/) can be used with either ArcGIS Desktop or ArcGIS Pro, and contains tools that perform geoprocessing tasks useful for working with Synthetic Aperture Radar (SAR) data. The tools were designed to be used with [Sentinel-1 Radiometric Terrain Corrected (RTC) SAR datasets](https://hyp3-docs.asf.alaska.edu/guides/rtc_product_guide/), such as those available on-demand using ASF's [Data Search - Vertex](https://hyp3-docs.asf.alaska.edu/using/vertex/) portal, but several of the tools have the potential to be used with a variety of rasters, including non-SAR datasets.
https://hyp3-docs.asf.alaska.edu/using/vertex/
The Toolbox is distributed as a zipped archive including the .pyt Toolbox script and associated .xml files. There is an XML file for the toolbox itself and one for each of the tools it contains. These XML files contain the metadata displayed in the item descriptions and tool help windows in ArcGIS, and must be kept in the same directory as the Python Toolbox (.pyt) file, or the information they contain will no longer be accessible.

### Toolbox Contents

#### Unzip Files Tool 
This tool assists in file management when downloading .zip files from ASF. It could be used to extract to a specified location any zip files with an additional internal directory containing the individual files. The tool deletes the original zip files once they are extracted, and is especially helpful when dealing with file paths that are so long that they are beyond the maximum allowed in default Windows unzip utilities.

#### Scale Conversion Tool 
This tool converts pixel values in calibrated SAR datasets (such as RTC rasters) from power, amplitude or dB scale into power, amplitude or dB scale. This is an application specific to SAR data values/scales.

#### Reclassify RTC Tool
This tool generates a raster that includes only those pixels below a user-defined threshold value, and is designed for isolating water pixels. While intended for RTC files in dB scale, this tool could be used for any application where the user is interested in generating a spatial mask for values below a given threshold in a single-band raster.

#### Log Difference Tool 
This tool compares two rasters by calculating the log difference on a pixel-by-pixel basis to identify areas where backscatter values have changed over time. While intended for RTC files in amplitude scale, this tool could be used to compare the pixel values of any two single-band rasters, as long as there are no negative values (NoData values will be returned for pixels with a negative number in either of the datasets).

#### RGB Decomposition Tool
This tool generates an RGB image using the co- and cross-polarized datasets from an RTC product. Input datasets can be in power, amplitude or dB scale, and the primary polarization can be either vertical (VV/VH) or horizontal (HH/HV). [Additional documentation](https://github.com/ASFHyP3/hyp3-lib/blob/develop/docs/rgb_decomposition.md) is available regarding the calculations used and the interpretation of these false-color images.

### Prerequisites
Users must have either ArcGIS Desktop (ArcMap) or ArcGIS Pro installed and licensed on their computer. The Toolbox has been tested with Desktop versions 10.6.1 and 10.7.1 and Pro versions 2.4.2, 2.5.x and 2.6.1, but it may work with earlier versions as well.

Note that several of the tools require the Spatial Analyst extension. Users who do not have licensing for this extension in ArcGIS will not be able to use many of the included tools.

### To install the Toolbox
- Download the zip file and extract the contents to any directory accessible by the computer running ArcGIS.
- Ensure that the Spatial Analyst extension is licensed and enabled.

    ##### ArcGIS Desktop (ArcMap) 
    - Click on the Customize menu in ArcMap and select Extensions… 
    - Check the box next to Spatial Analyst and click the Close button at the bottom of the Extensions window. 
        - If you are unable to check this box, you do not have access to the Spatial Analyst extension and will not be able to make use of tools requiring this extension.
        
    ##### ArcGIS Pro  
    - Click on the Project tab and select the Licensing tab. 
    - In the list of Esri Extensions, scroll down to verify that the Spatial Analyst is licensed and enabled. 
        - If it is not, an organization administrator will need to enable the extension in your user account. 
        - If your organization does not have a license available for you to use, you will not be able to make use of tools requiring this extension.

### Using the Toolbox
In the ArcMap Catalog window or the ArcGIS Pro Catalog pane/view, navigate to the directory containing the toolbox (create a new folder connection if necessary).
- To open the Catalog window in ArcMap, click on the Windows menu and select Catalog.
- To open the Catalog pane or view in ArcGIS Pro, click the View tab and click on either the Catalog Pane or Catalog View button.
     
*Note that if you explore the extracted contents of the zip file outside of the ArcGIS environment, the directory will contain one .pyt file and a number of .xml files.*
 
 In the ArcGIS Catalog window/pane/view, only the Toolbox is displayed, and when it is expanded, all of the Tools contained in the Toolbox script are displayed. The XML files are automatically referenced when ArcGIS requires the information they contain, and do not appear as additional files in the ArcGIS Catalog environment. The XML files must remain in the same directory as the .pyt file, and their filenames should not be changed.
 
- Double-click the ASF_Tools.pyt file to display the Tools (Scripts) included in the toolbox.
- Double-click on a Tool (displayed with a Script icon) to launch the dialog box or geoprocessing pane, as you would for any other ArcGIS Tool/Script.
- Enter the parameters as prompted and click the OK button to execute the tool.

*Note that output products are not automatically added to a project by default. You must navigate to them in the Catalog window/pane/view (or using the Add Data dialog) and add them to your project if desired.*

### Tool Help
The XML files included in the zip file are accessed when a user views the metadata for the toolbox, individual tools, or even different fields within the tool dialog.

#### Accessing Help from within the Tool Dialog Box

##### ArcGIS Desktop
- Click on the Show Help button at the bottom of the tool window to open the help panel. 
    - This panel will display information about the tool in general if no field is activated. 
    - If the user clicks on any of the parameter fields, information specific to that parameter will be displayed.
- Click on the Tool Help button at the bottom of the Help pane to open another window that displays most of the information that would be displayed in the tool’s Item Description. 

##### ArcGIS Pro
- When you hover over any of the parameter fields in the tool dialog, a blue i appears. Hover over or click the blue i icon to view helpful tips specific to that parameter.
- Hover over the blue question mark at the top of the geoprocessing pane to display information about the tool. Click on it to open the full tool description in a browser window. 

#### Accessing Help from the Catalog Interface

##### ArcGIS Desktop
ArcCatalog displays the information contained in the xml metadata files in the Description tab for the toolbox and each tool.

In the ArcMap Catalog window, the Item Description for the toolbox or any of its constituent tools displays the xml content.
- Right-click the toolbox or tool in the Catalog window and select Item Description to view the information.

##### ArcGIS Pro
The xml metadata is displayed in the Metadata tab in the Catalog view.
- Right-click a tool in the Catalog pane and select View Metadata to open the Metadata tab for the item in the Catalog view.  
    **_OR_**
- Open the Catalog View directly to navigate to the tool and select the Metadata tab.
