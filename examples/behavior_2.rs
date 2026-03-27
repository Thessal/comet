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
                    "behavior_2_compare.pth",
                ),
                train: Some(
                    true,
                ),
                supervised_epochs: Some(
                    100,
                ),
                operators: None,
                integers: None,
                floats: None,
                strings: None,
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
                    Assignment {
                        target: "eps_levels",
                        expr: List(
                            [
                                Literal(
                                    Float(
                                        0.1,
                                    ),
                                ),
                                Literal(
                                    Float(
                                        0.5,
                                    ),
                                ),
                                Literal(
                                    Float(
                                        1.0,
                                    ),
                                ),
                            ],
                        ),
                    },
                    Assignment {
                        target: "x",
                        expr: Call {
                            path: Path {
                                segments: [
                                    "Comparator",
                                ],
                            },
                            args: [
                                Identifier(
                                    "volume",
                                ),
                                Identifier(
                                    "adv20",
                                ),
                                Identifier(
                                    "eps_levels",
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
                                    "ts_diff",
                                ],
                            },
                            args: [
                                Identifier(
                                    "x",
                                ),
                                Identifier(
                                    "variousdays",
                                ),
                            ],
                        },
                    ),
                ],
            },
        ),
    ],
}