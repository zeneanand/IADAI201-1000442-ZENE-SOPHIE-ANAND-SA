from src.insights import build_insights, congestion_level
def test_congestion_thresholds():
    assert congestion_level(10) == "Low"
    assert congestion_level(40) == "Moderate"
    assert congestion_level(75) == "Moderate"
    assert congestion_level(76) == "High"
def test_counts_and_recommendation():
    labels = ["occupied"] * 9 + ["empty"] * 3
    insights = build_insights(labels)
    assert insights.total_slots == 12
    assert insights.occupied_slots == 9
    assert insights.available_slots == 3
    assert insights.occupancy_pct == 75.0
    assert insights.congestion == "Moderate"
