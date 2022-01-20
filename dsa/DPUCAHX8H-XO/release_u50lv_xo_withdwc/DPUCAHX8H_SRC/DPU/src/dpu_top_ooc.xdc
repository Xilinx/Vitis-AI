create_clock -name ACLK      -period 3.332 [get_ports ACLK]
create_clock -name ACLK_DR   -period 1.666 [get_ports ACLK_DR]
create_clock -name ACLK_REG  -period 10    [get_ports s_axi_clk]
