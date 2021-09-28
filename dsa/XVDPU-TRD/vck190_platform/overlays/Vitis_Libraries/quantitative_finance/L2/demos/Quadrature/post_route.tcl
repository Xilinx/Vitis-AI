#set unroute_nets [report_route_status -route_type UNROUTE -return_nets]

#route_design -nets [get_nets $unroute_nets]
