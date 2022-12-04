
# vai_add_test(  tf2_custom_op_graph_runner  REQUIRE vitis_ai_library::graph_runner vart::runner xir)
vai_add_sample(NAME tf2_custom_op_graph_runner
               SRCS tf2_custom_op_graph_runner.cpp
	       PUBLIC_REQUIRE  vitis_ai_library::graph_runner vart::runner
        )

