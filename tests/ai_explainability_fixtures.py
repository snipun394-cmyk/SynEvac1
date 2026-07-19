from ai_training.models.bottleneck_model import BottleneckModel
from ai_training.models.evacuation_time_model import EvacuationTimeModel
from ai_training.split import apply_split, make_split

from tests.ai_training_fixtures import RealCampaignTestCase


class RealTrainedModelsTestCase(RealCampaignTestCase):

    # Extends RealCampaignTestCase (a real generated campaign, built
    # once per test class) with two already-fitted real models -- a
    # regression model (EvacuationTimeModel) and a classification model
    # (BottleneckModel, target="location") -- shared read-only across
    # every ai_explainability test that needs something already trained
    # to explain/benchmark/compare. ai_explainability never fits a
    # model itself in production code; only these test fixtures do, to
    # set up realistic inputs.

    @classmethod
    def setUpClass(cls):

        super().setUpClass()

        X, y, _extra = EvacuationTimeModel.build_table(cls.dataset)
        split = make_split(len(X), test_size=0.3, random_state=0)
        cls.evac_X_train, _val, cls.evac_X_test = apply_split(X, split)
        cls.evac_y_train, _val, cls.evac_y_test = apply_split(y, split)

        cls.evac_model = EvacuationTimeModel()
        cls.evac_model.fit(cls.evac_X_train, cls.evac_y_train)

        X_b, y_b, _extra = BottleneckModel.build_table(cls.dataset, target="location")
        split_b = make_split(len(X_b), test_size=0.3, random_state=0)
        cls.bottleneck_X_train, _val, cls.bottleneck_X_test = apply_split(X_b, split_b)
        cls.bottleneck_y_train, _val, cls.bottleneck_y_test = apply_split(y_b, split_b)

        cls.bottleneck_model = BottleneckModel(target="location")
        cls.bottleneck_model.fit(cls.bottleneck_X_train, cls.bottleneck_y_train)
