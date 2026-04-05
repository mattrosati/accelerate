TARGETS = [
    "MAPopt_Yale_affected_beta",
    "LLA_Yale_affected_beta",
    "ULA_Yale_affected_beta",
    "Yale_R2full_affected",
]

FEATURES = ["hr", "rso2r", "rso2l", "spo2", "abp"]

PERCENT_IN_MIN = 0.5
PERCENT_NA_MAX = 0.25
WINDOW_SECONDS = 60

SMOOTH_FRAC_OUT_MIN = 0.46

AR_CLASSES = {0: "below", 1: "in", 2: "above"}

ABP_PHYSIO_LO = 20.0
ABP_PHYSIO_HI = 200.0
ABP_MAX_BAD_FRAC = 0.25
