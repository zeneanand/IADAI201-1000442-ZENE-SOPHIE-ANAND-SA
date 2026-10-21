"""Convert slot predictions into parking availability insights."""
from dataclasses import dataclass
LOW_THRESHOLD = 40.0
HIGH_THRESHOLD = 75.0
@dataclass
class ParkingInsights:
    total_slots: int
    occupied_slots: int
    available_slots: int
    occupancy_pct: float
    congestion: str
    recommendation: str
    color: str
def occupancy_percentage(occupied: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((occupied / total) * 100.0, 1)
def congestion_level(occupancy_pct: float) -> str:
    if occupancy_pct < LOW_THRESHOLD:
        return "Low"
    if occupancy_pct <= HIGH_THRESHOLD:
        return "Moderate"
    return "High"
def recommendation_for(level: str) -> tuple[str, str]:
    if level == "High":
        return (
            "Parking full — try another area",
            "#EF4444",
        )
    if level == "Moderate":
        return (
            "Some slots available — proceed, but expect limited space",
            "#F59E0B",
        )
    return (
        "Slots available — proceed",
        "#22C55E",
    )
def build_insights(labels: list[str]) -> ParkingInsights:
    total = len(labels)
    occupied = sum(1 for label in labels if label == "occupied")
    available = total - occupied
    pct = occupancy_percentage(occupied, total)
    level = congestion_level(pct)
    rec, color = recommendation_for(level)
    return ParkingInsights(
        total_slots=total,
        occupied_slots=occupied,
        available_slots=available,
        occupancy_pct=pct,
        congestion=level,
        recommendation=rec,
        color=color,
    )
