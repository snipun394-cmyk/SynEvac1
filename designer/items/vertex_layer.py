from designer.items.vertex_handle import VertexHandle


class VertexLayer:

    def __init__(self, scene):

        self.scene = scene
        self.handles = []

        self.current_item = None

    # =====================================================

    def clear(self):

        for handle in self.handles:

            if handle.scene():

                handle.scene().removeItem(handle)

        self.handles.clear()

        self.current_item = None

    # =====================================================

    def show(self, item):

        self.clear()

        self.current_item = item

        polygon = item.polygon()

        for i in range(polygon.count()):

            point = polygon.at(i)

            handle = VertexHandle(
                point.x(),
                point.y(),
                i,
                item,
            )

            self.scene.addItem(handle)

            self.handles.append(handle)

    # =====================================================

    def hide(self):

        self.clear()

    # =====================================================

    def refresh(self):

        if self.current_item:

            self.show(self.current_item)