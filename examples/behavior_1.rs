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