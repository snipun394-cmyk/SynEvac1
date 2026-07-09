class SelectionManager:

    def __init__(self):

        self.selected_item = None

        self.selection_changed_callback = None

    # =====================================================

    def clear(self):

        if self.selected_item:

            self.selected_item.set_selected(False)

        self.selected_item = None

        if self.selection_changed_callback:

            self.selection_changed_callback(None)

    # =====================================================

    def select(self, item):

        if self.selected_item:

            self.selected_item.set_selected(False)

        self.selected_item = item

        if self.selected_item:

            self.selected_item.set_selected(True)

        if self.selection_changed_callback:

            self.selection_changed_callback(
                self.selected_item
            )

    # =====================================================

    def selected(self):

        return self.selected_item

    # =====================================================

    def has_selection(self):

        return self.selected_item is not None

    # =====================================================

    def delete_selected(self):

        item = self.selected_item

        self.clear()

        return item