from mlwkf.constants import NON_COVARIATES_FIELDS


class TestNonCovariatesFields:

    def test_contains_required_fields(self):
        required = {"x", "y", "target", "weight", "groupcv", "groupcv_class"}
        assert required.issubset(set(NON_COVARIATES_FIELDS))

    def test_is_list(self):
        assert isinstance(NON_COVARIATES_FIELDS, list)

    def test_no_duplicates(self):
        assert len(NON_COVARIATES_FIELDS) == len(set(NON_COVARIATES_FIELDS))
