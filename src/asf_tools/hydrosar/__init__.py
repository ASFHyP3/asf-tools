from warnings import warn

HYDROSAR_MOVE_WARNING = \
    """
    ---------------------------------------------------------------------------
    The HydroSAR codes (`flood_map`, `water_map` and `hand` modules) are being
    moved to the HydroSAR project repository:
        <https://github.com/fjmeyer/hydrosar>
    and will be provided in a new pip/conda installable package `hydrosar`.

    The `asf_tools.hydrosar` subpackage will be removed in a future release.
    ----------------------------------------------------------------------------
    """

warn(HYDROSAR_MOVE_WARNING,  category=FutureWarning)
