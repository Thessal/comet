Program {
    declarations: [
        Behavior(
            BehaviorDecl {
                name: "Comparator",
                args: [
                    TypedArg {
                        name: "signal",
                        type_decl: DataFrame,
                    },
                    TypedArg {
                        name: "eps",
                        type_decl: Float,
                    },
                    TypedArg {
                        name: "reference",
                        type_decl: DataFrame,
                    },
                ],
                return_type: DataFrame,
                weights: Some(
                    "behavior_1_compare.pth",
                ),
                train: Some(
                    true,
                ),
                supervised_epochs: Some(
                    100,
                ),
                operators: Some(
                    [
                        "add",
                        "divide",
                        "ts_mean",
                        "ts_diff",
                        "consume_float",
                        "cs_rank",
                        "ts_rank",
                        "cs_zscore",
                        "ts_zscore",
                    ],
                ),
                integers: Some(
                    [
                        5,
                        21,
                        252,
                    ],
                ),
                floats: Some(
                    [
                        0.1,
                        0.5,
                        0.9,
                        5.0,
                        21.0,
                        252.0,
                    ],
                ),
                strings: Some(
                    [
                        "volume",
                        "adv20",
                    ],
                ),
            },
        ),
        Flow(
            FlowDecl {
                name: "volume_spike",
                body: [
                    Assignment {
                        target: "volume",
                        expr: Call {
                            path: Path {
                                segments: [
                                    "data",
                                ],
                            },
                            args: [
                                Literal(
                                    String(
                                        "volume",
                                    ),
                                ),
                            ],
                        },
                    },
                    Assignment {
                        target: "adv20",
                        expr: Call {
                            path: Path {
                                segments: [
                                    "data",
                                ],
                            },
                            args: [
                                Literal(
                                    String(
                                        "adv20",
                                    ),
                                ),
                            ],
                        },
                    },
                    Expr(
                        Call {
                            path: Path {
                                segments: [
                                    "Comparator",
                                ],
                            },
                            args: [
                                Identifier(
                                    "volume",
                                ),
                                Literal(
                                    Float(
                                        0.1,
                                    ),
                                ),
                                Identifier(
                                    "adv20",
                                ),
                            ],
                        },
                    ),
                ],
            },
        ),
    ],
}