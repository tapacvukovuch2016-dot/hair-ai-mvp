"""
Photo validation logic for Hair AI MVP v0.2.

This module does NOT run computer vision itself yet.
It defines the contract expected from a future vision model and converts
its observations into actionable requests for the user.
"""

REQUIRED_FRONT = {
    "face_visible": True,
    "hair_visible": True,
    "no_headwear": True,
    "lighting_ok": True,
    "not_blurry": True,
}

def validate_photo(observation: dict) -> dict:
    issues = []
    requests = []
    usable_for = []

    if not observation.get("face_visible", False):
        issues.append("face_not_visible")
        requests.append("Сделай фото, где лицо полностью видно.")
    else:
        usable_for.append("face_proportions")

    if not observation.get("hair_visible", False):
        issues.append("hair_not_visible")
        requests.append("Покажи волосы полностью, без капюшона или других перекрытий.")
    else:
        usable_for.append("visible_hair_features")

    if observation.get("headwear", False):
        issues.append("headwear")
        requests.append("Сними шапку, кепку или другой головной убор.")

    if observation.get("hair_tied", False):
        issues.append("hair_tied")
        requests.append("Если возможно, распусти волосы и сделай дополнительное фото.")

    if not observation.get("lighting_ok", False):
        issues.append("poor_lighting")
        requests.append("Сделай фото при более равномерном освещении.")

    if observation.get("blurry", False):
        issues.append("blurry")
        requests.append("Сделай более чёткое фото без движения камеры.")

    angle = observation.get("angle", "unknown")
    if angle == "front":
        usable_for += ["face_shape", "hairline"]
    elif angle in ("left_profile", "right_profile"):
        usable_for += ["profile", "partial_head_shape"]
    elif angle == "back":
        usable_for += ["occipital_shape", "crown", "back_density"]

    confidence = float(observation.get("overall_confidence", 0))
    if confidence < 0.40:
        decision = "reject"
    elif issues:
        decision = "partial"
    else:
        decision = "accept"

    return {
        "decision": decision,
        "issues": issues,
        "usable_for": sorted(set(usable_for)),
        "next_requests": requests,
        "overall_confidence": confidence,
    }


def determine_missing_views(session: dict) -> list:
    """Determine which views are still useful after all uploaded photos."""
    covered = set()
    for photo in session.get("photos", []):
        result = validate_photo(photo)
        if result["decision"] != "reject":
            covered.update(result["usable_for"])

    requests = []
    if "face_shape" not in covered:
        requests.append("Нужно фронтальное фото.")
    if "profile" not in covered:
        requests.append("Нужно фото в профиль.")
    if "occipital_shape" not in covered:
        requests.append("Для более точной формы головы желательно фото сзади.")
    return requests


if __name__ == "__main__":
    examples = json.load(open("photo_test_cases.json", encoding="utf-8"))
    for name, observation in examples.items():
        print("\\n===", name, "===")
        print(json.dumps(validate_photo(observation), ensure_ascii=False, indent=2))
