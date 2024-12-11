# NOTE: Double check before using

SCORING_SCHEMAS = {
    "bdi-ii": {
        "components": {
            "total": range(1, 22),
        },
        "alert_threshold": 14,
        "alert_message": "depression",
    },
    "brs": {
        "components": {
            "total": range(1, 7),
        },
        "reverse_items": [2, 4, 6],
        "min_item_score": 1,
        "max_item_score": 5,
        "metric": "mean",
    },
    "cd-risc-25": {
        "components": {
            "total": range(1, 26),
        },
    },
    "cd-risc-10": {
        "components": {
            "total": range(1, 11),
        },
    },
    "erq": {
        "components": {
            "reappraisal": [1, 3, 5, 7, 8, 10],
            "suppression": [2, 4, 6, 9],
        },
        "metric": "mean",
    },
    "ffmq-d": {
        "components": {
            "nonjudging": [3, 10, 14, 17, 25, 30, 35, 39],
            "describing": [2, 7, 12, 16, 22, 27, 32, 37],
            "observing": [1, 6, 11, 15, 20, 26, 31, 36],
            "acting_with_awareness": [5, 8, 13, 18, 23, 28, 34, 38],
            "nonreactivity": [4, 9, 19, 21, 24, 29, 33],
        },
        "reverse_items": [
            3,
            10,
            14,
            17,
            25,
            30,
            35,
            39,
            12,
            16,
            22,
            5,
            8,
            13,
            18,
            23,
            28,
            34,
            38,
        ],
        "min_item_score": 1,
        "max_item_score": 6,
    },
    "fmi-14": {
        "components": {
            "total": range(1, 15),
        },
        "reverse_items": [13],
        "min_item_score": 1,
        "max_item_score": 4,
    },
    "gad-7": {
        "components": {
            "total": range(1, 8),
        },
        "alert_threshold": 10,
        "alert_message": "a generalized anxiety disorder",
    },
    "iri-s": {
        "components": {
            "fantasy": [2, 7, 12, 15],
            "empathic_concern": [1, 5, 9, 11],
            "perspective_taking": [4, 10, 14, 16],
            "personal_distress": [3, 6, 8, 13],
        },
    },
    "lot-r": {
        "components": {
            # following two-dimensional model of indepentent optimism and pessimism
            "pessimism": [3, 7, 9],
            "optimism": [1, 4, 10],
        },
        "filler_items": [2, 5, 6, 8],
    },
    "maia-2": {
        "components": {
            "noticing": [1, 2, 3, 4],
            "not_distracting": [5, 6, 7, 8, 9, 10],  # items are reverse-scored
            "not_worrying": [11, 12, 13, 14, 15],  # some items are reverse-scored
            "attention_regulation": [16, 17, 18, 19, 20, 21, 22],
            "emotional_awareness": [23, 24, 25, 26, 27],
            "self_regulation": [28, 29, 30, 31],
            "body_listening": [32, 33, 34],
            "trusting": [35, 36, 37],
        },
        "reverse_items": [5, 6, 7, 8, 9, 10, 11, 12, 15],
        "min_item_score": 0,
        "max_item_score": 5,
    },
    "maas": {
        "components": {
            "total": range(1, 16),
        },
        "metric": "mean",
    },
    "panas": {
        "components": {
            "positive_affect": [1, 3, 4, 6, 10, 11, 13, 15, 17, 18],
            "negative_affect": [2, 5, 7, 8, 9, 12, 14, 16, 19, 20],
        },
        "metric": "mean",
    },
    "pcs": {
        "components": {
            "total": range(1, 14),
            "rumination": [8, 9, 10, 11],
            "magnification": [6, 7, 13],
            "helplessness": [1, 2, 3, 4, 5, 12],
        },
        "alert_threshold": 30,
        "alert_message": "a clinically significant level of catastrophizing",
    },
    "phq-15": {
        "components": {
            "total": range(1, 16),
        },
        "metric": "sum",
        "alert_threshold": 10,
        "alert_message": "a clinically significant level of somatic symptoms",
    },
    "pvaq": {
        "components": {
            "total": range(1, 17),
            "attention_to_pain": [1, 6, 7, 8, 10, 12, 13, 14, 15, 16],
            "attention_to_changes": [2, 3, 4, 5, 9, 11],
        },
        "reverse_items": [8, 16],
        "min_item_score": 0,
        "max_item_score": 5,
    },
    "stai-t-10": {
        "components": {
            "total": range(1, 11),
        },
        "reverse_items": [3, 4, 7],
        "min_item_score": 1,
        "max_item_score": 8,
        "metric": "percentage",
    },
}
