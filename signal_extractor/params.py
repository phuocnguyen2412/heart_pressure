params = {
    "extractors": [
        {
            "functions": ["luma_component_mean"],
            "name": "luma_mean",
            "parameters": {"initial_skip_seconds": 0},
        },
        {
            "functions": ["red_channel_mean"],
            "name": "r_ch_mean",
            "parameters": {"initial_skip_seconds": 0},
        },
    ],
    "preprocessor": {
        "filter_chains": [
            {
                "flist": [
                    {"name": "roll_avg", "params": {"window_size_seconds": 1.01}},
                    {"name": "sub", "params": {}},
                    {"name": "lpf", "params": {"filter_order": 2, "low": 4}},
                    {"name": "cut_start", "params": {"seconds": 0}},
                ],
                "name": "chain2",
            },
            {
                "flist": [
                    {"name": "cut_start", "params": {"seconds": 0}},
                    {"name": "hpf", "params": {"cutoff": 0.5, "order": 1}},
                    {
                        "name": "bpf_bpm",
                        "params": {"mincut": 0.01, "multiplier": 3, "order": 1},
                    },
                ],
                "name": "dynamic_bpm",
            },
        ],
        "sources": ["r_ch_mean"]
    },
}
