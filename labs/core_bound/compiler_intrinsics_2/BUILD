[
    (
        cc_library(
            name = "solution" + postfix,
            srcs = [
                # "init.cpp",
                "solution" + postfix + ".cpp",
            ],
            hdrs = ["solution.hpp"],
        ),
        cc_binary(
            name = "bench" + postfix,
            srcs = [
                "bench.cpp",
            ],
            deps = [
                ":solution" + postfix,
                "@benchmark",
            ],
        ),
        cc_binary(
            name = "validate" + postfix,
            srcs = [
                "validate.cpp",
            ],
            deps = [
                ":solution" + postfix,
            ],
        ),
    )
    for postfix in [
        "",
        "_ptbs",
    ]
]
