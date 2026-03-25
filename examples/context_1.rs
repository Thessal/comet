Program {
    declarations: [
        Behavior(
            BehaviorDecl {
                name: "Compare",
                args: [
                    TypedArg {
                        name: "signal",
                        type_decl: DataFrame,
                    },
                    TypedArg {
                        name: "reference",
                        type_decl: DataFrame,
                    },
                ],
                return_type: DataFrame,
                weights: Some(
                    "context_1_compare.pth",
                ),
                train: Some(
                    false,
                ),
                supervised_samples: Some(
                    100,
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
                        target: "variousdays",
                        expr: List(
                            [
                                Literal(
                                    Integer(
                                        1,
                                    ),
                                ),
                                Literal(
                                    Integer(
                                        2,
                                    ),
                                ),
                                Literal(
                                    Integer(
                                        3,
                                    ),
                                ),
                                Literal(
                                    Integer(
                                        4,
                                    ),
                                ),
                                Literal(
                                    Integer(
                                        5,
                                    ),
                                ),
                                Literal(
                                    Integer(
                                        21,
                                    ),
                                ),
                                Literal(
                                    Integer(
                                        252,
                                    ),
                                ),
                            ],
                        ),
                    },
                    Expr(
                        Call {
                            path: Path {
                                segments: [
                                    "Compare",
                                ],
                            },
                            args: [
                                Identifier(
                                    "volume",
                                ),
                                Call {
                                    path: Path {
                                        segments: [
                                            "ts_mean",
                                        ],
                                    },
                                    args: [
                                        Identifier(
                                            "volume",
                                        ),
                                        Identifier(
                                            "variousdays",
                                        ),
                                    ],
                                },
                            ],
                        },
                    ),
                ],
            },
        ),
    ],
}