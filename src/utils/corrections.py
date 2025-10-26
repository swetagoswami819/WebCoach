def check_form(exercise, angles, sensitivity=1.0):
    """
    Given exercise name and angles dict (from PoseEstimator.compute_angles),
    return a list of strings with prioritized hints (most important first).
    The correctness rules are simple and rule-based — tweak thresholds by `sensitivity`.
    """
    hints = []

    # safety: if no angles available return empty
    if not angles:
        return hints

    # helper thresholds (lower sensitivity => more tolerant)
    def thresh(x):
        return x * (1.0 / sensitivity)

    # --- SQUAT checks
    if exercise == "Squat":
        # Knee valgus (knees caving in): check knee_center_dist small relative to ankle (approx)
        left_knee = angles.get("left_knee_angle", 180)
        right_knee = angles.get("right_knee_angle", 180)
        left_knee_dx = angles.get("left_knee_ankle_dx", 0.0)
        right_knee_dx = angles.get("right_knee_ankle_dx", 0.0)

        # Back angle (if back_angle small => torso too horizontal)
        back_ang = angles.get("back_angle", 180)

        # Rule: knees should track ankles (knee_ankle_dx close to 0). If dx is negative and large -> knees inward
        if left_knee_dx < -0.03 or right_knee_dx < -0.03:
            hints.append("Knees are caving inward — push knees out (avoid valgus).")
        # Rule: back_angle should not be too small (avoid excessive forward lean)
        if back_ang < thresh(40):
            hints.append("Keep your chest up — don't lean forward too much.")
        # Rule: depth — check knee angle too shallow
        avg_knee = (left_knee + right_knee) / 2.0
        if avg_knee > thresh(140):
            hints.append("Go deeper for a full squat (knees more bent).")
        # Rule: if knees go too far forward over toes (approx using knee dx positive large)
        if left_knee_dx > 0.12 or right_knee_dx > 0.12:
            hints.append("Knees are too far forward — shift hips back.")

    # --- PUSH-UP checks
    if exercise == "Push-up":
        # hip sag: using back_angle (if too large => hips sagging)
        back_ang = angles.get("back_angle", 180)
        left_elbow = angles.get("left_elbow_angle", 180)
        right_elbow = angles.get("right_elbow_angle", 180)
        avg_elbow = (left_elbow + right_elbow) / 2.0

        if back_ang > thresh(30):
            hints.append("Hips sagging — keep a straight line from shoulders to ankles.")
        if avg_elbow > thresh(150):
            hints.append("Lower further — bend elbows to ~90° for full depth.")
        if avg_elbow < thresh(30):
            hints.append("Don't lock elbows at the top — control your movement.")

    # --- DEADLIFT checks
    if exercise == "Deadlift":
        back_ang = angles.get("back_angle", 180)
        left_knee = angles.get("left_knee_angle", 180)
        right_knee = angles.get("right_knee_angle", 180)
        avg_knee = (left_knee + right_knee) / 2.0

        # Rule: keep neutral spine — back_angle should be around 40-80 depending on stance
        if back_ang < thresh(20):
            hints.append("Don't round your lower back — keep spine neutral.")
        if avg_knee < thresh(30):
            hints.append("Too much knee bend — drive with hips to target hamstrings/glutes.")

    # If no hints -> positive reinforcement
    if not hints:
        hints.append("Good form — keep it up!")

    return hints
