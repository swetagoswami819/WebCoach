class RepCounter:
    """
    Generic simple rep counter using angle thresholds.
    - down_angle: angle less-than threshold indicates "down" position (e.g., squat bottom)
    - up_angle: angle greater-than threshold indicates "up" position (standing)
    State machine: up -> down -> up counts as 1 rep.
    """

    def __init__(self, exercise="Squat", down_angle=80, up_angle=160):
        self.exercise = exercise
        self.down_angle = down_angle
        self.up_angle = up_angle
        self.state = "up"  # start assuming up
        # small hysteresis
        self.down_confirm = False

    def update(self, angle):
        """
        angle: float or None
        returns "none", "up_to_down", or "down_to_up"
        """
        if angle is None:
            return None
        # ensure numeric
        try:
            ang = float(angle)
        except Exception:
            return None

        if self.state == "up":
            if ang < self.down_angle:
                self.state = "down"
                return "up_to_down"
        elif self.state == "down":
            if ang > self.up_angle:
                self.state = "up"
                return "down_to_up"
        return None
