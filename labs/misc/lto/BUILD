cc_library(
    name = "ao",
    srcs = [
        "ao.cpp",
        "ao_helpers.cpp",
        "ao_init.cpp",
        "ao_intersect.cpp",
        "ao_occlusion.cpp",
        "ao_orthoBasis.cpp",
        "ao_render.cpp",
    ],
    hdrs = ["ao.h"],
)

cc_binary(
    name = "bench",
    srcs = [
        "bench.cpp",
    ],
    deps = [
        ":ao",
        "@benchmark",
    ],
)

cc_binary(
    name = "validate",
    srcs = [
        "validate.cpp",
    ],
    deps = [
        ":ao",
    ],
)
