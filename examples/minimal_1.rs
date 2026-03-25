Program {
    declarations: [
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
                        target: "mean_vol",
                        expr: Call {
                            path: Path {
                                segments: [
                                    "ts_mean",
                                ],
                            },
                            args: [
                                Identifier(
                                    "volume",
                                ),
                                Literal(
                                    Integer(
                                        10,
                                    ),
                                ),
                            ],
                        },
                    },
                    Expr(
                        Call {
                            path: Path {
                                segments: [
                                    "divide",
                                ],
                            },
                            args: [
                                Identifier(
                                    "volume",
                                ),
                                Identifier(
                                    "mean_vol",
                                ),
                            ],
                        },
                    ),
                ],
            },
        ),
    ],
}