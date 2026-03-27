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
                supervised_samples: Some(
                    10,
                ),
                operators: Some(
                    [
                        "add",
                        "divide",
                        "ts_mean",
                        "cs_rank",
                        "ts_diff",
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