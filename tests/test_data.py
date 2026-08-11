from steerfrontier.data import load_generation_data


def test_sycophancy_generation_dataset():
    data = load_generation_data("data/sycophancy/generate.json")

    assert len(data) == 1000
    assert data[0]["answer_matching_behavior"] in {"(A)", "(B)"}
    assert data[0]["answer_not_matching_behavior"] in {"(A)", "(B)"}
