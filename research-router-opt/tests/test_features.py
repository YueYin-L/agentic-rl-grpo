import pytest

from research_router_opt.features import FEATURE_NAMES, extract_features


def test_extract_features_has_stable_dimension() -> None:
    features = extract_features("比较两篇雪豹论文的实验结论")
    assert len(features) == len(FEATURE_NAMES)
    assert features[0] == 1.0
    assert features[2] >= 1.0
    assert features[3] >= 1.0


def test_extract_features_rejects_blank_query() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        extract_features("   ")

