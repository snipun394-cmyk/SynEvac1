import math

from PyQt6.QtWidgets import QGridLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from command_center.camera_tile_widget import CameraTileWidget


class LiveCameraGridPanel(QWidget):

    # Live CCTV Dashboard milestone -- the Command Center tab hosting
    # one CameraTileWidget per camera the Building has configured
    # (command_center/live_camera_view_gateway.py::LiveCameraViewGateway.
    # camera_tiles()). "Dynamic grid" means exactly this: the number of
    # tiles AND the grid's row/column shape are both recomputed from
    # however many cameras camera_tiles() returns each refresh -- never
    # a fixed 4/8/N hardcoded here. Today that's one tile (Camera 1);
    # configuring more Live cameras grows the grid automatically, no
    # code change required.
    #
    # Tiles are created lazily and reused across ticks -- rebuilt only
    # when the actual SET of camera_ids changes (a camera added/removed
    # from the Building this session), never on an ordinary ~1Hz
    # refresh, so this stays cheap regardless of camera count.

    def __init__(self):
        super().__init__()

        self._tiles = {}  # camera_id -> CameraTileWidget
        self._tile_order = ()

        outer = QVBoxLayout(self)

        self._empty_label = QLabel("No live camera session active.")
        outer.addWidget(self._empty_label)

        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)

        # Live CCTV Layout milestone -- the grid's own natural size
        # grows with however many cameras are configured (32 for the
        # real laboratory building, p3.syn) and can easily exceed the
        # Command Center window's actual height well before it exceeds
        # any per-tile minimum size. A QScrollArea is the only change
        # needed: every tile keeps its existing minimum size/aspect
        # ratio/rendering untouched (CameraTileWidget itself is not
        # modified at all), the grid's own column math is unchanged,
        # and this scales to any future camera count with zero further
        # code changes -- scrolling further replaces clipping, nothing
        # else about the panel's behavior changes. setWidgetResizable
        # (True) lets the scroll area's VIEWPORT track the panel's own
        # available space while the CONTENT (_grid_container) keeps
        # whatever size its own layout naturally computes -- exactly
        # the "shrink the viewport, not the content" behavior needed
        # here (the reverse, letting the content shrink, is what
        # caused today's clipping in the first place).
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setWidget(self._grid_container)
        outer.addWidget(self._scroll_area, 1)

        self._scroll_area.setVisible(False)

    # =====================================================
    # Public entry point
    # =====================================================

    def show_live(self, gateway) -> None:

        if gateway is None:

            self._empty_label.setVisible(True)
            self._scroll_area.setVisible(False)
            return

        tiles_data = gateway.camera_tiles()
        camera_ids = tuple(tile.camera_id for tile in tiles_data)

        if camera_ids != self._tile_order:
            self._rebuild_grid(camera_ids)

        for tile_data in tiles_data:
            self._tiles[tile_data.camera_id].refresh(tile_data)

        has_cameras = len(tiles_data) > 0
        self._empty_label.setVisible(not has_cameras)
        self._scroll_area.setVisible(has_cameras)

    # =====================================================
    # Grid shape: a near-square layout sized off however many cameras
    # exist THIS refresh -- e.g. 1 camera -> 1x1, 4 -> 2x2, 5-8 -> 3x3
    # (with empty trailing cells), never a hardcoded channel count.
    # =====================================================

    def _rebuild_grid(self, camera_ids) -> None:

        while self._grid_layout.count():

            item = self._grid_layout.takeAt(0)
            widget = item.widget()

            if widget is not None:
                widget.setParent(None)

        self._tiles = {}
        self._tile_order = camera_ids

        columns = max(1, math.ceil(math.sqrt(len(camera_ids)))) if camera_ids else 1

        for index, camera_id in enumerate(camera_ids):

            tile = CameraTileWidget()
            self._tiles[camera_id] = tile

            row, column = divmod(index, columns)
            self._grid_layout.addWidget(tile, row, column)
