from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QGraphicsPixmapItem


class FloorPlanItem(QGraphicsPixmapItem):

    def __init__(self, image_path: str):
        super().__init__()

        self.setPixmap(QPixmap(image_path))

        self.setZValue(-100)