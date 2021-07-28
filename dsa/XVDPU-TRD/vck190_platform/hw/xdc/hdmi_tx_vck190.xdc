# HDMI TX
set_property PACKAGE_PIN F41 [get_ports TX_DATA_OUT_txp[0]]
set_property PACKAGE_PIN D41 [get_ports TX_DATA_OUT_txp[1]]
set_property PACKAGE_PIN B41 [get_ports TX_DATA_OUT_txp[2]]
set_property PACKAGE_PIN A43 [get_ports TX_DATA_OUT_txp[3]]

set_property PACKAGE_PIN F46 [get_ports RX_DATA_IN_rxp[0]]
set_property PACKAGE_PIN E44 [get_ports RX_DATA_IN_rxp[1]]
set_property PACKAGE_PIN D46 [get_ports RX_DATA_IN_rxp[2]]
set_property PACKAGE_PIN C44 [get_ports RX_DATA_IN_rxp[3]]

#HDMI_8T49N241_OUT_C_P
set_property PACKAGE_PIN E39 [get_ports TX_REFCLK_P_IN]
#HDMI_TX_HPD
set_property PACKAGE_PIN K18 [get_ports TX_HPD_IN]
set_property IOSTANDARD LVCMOS33 [get_ports TX_HPD_IN]
#HDMI_TX_SRC_SCL
set_property PACKAGE_PIN K21 [get_ports TX_DDC_OUT_scl_io]
set_property IOSTANDARD LVCMOS33 [get_ports TX_DDC_OUT_scl_io]
#HDMI_TX_SRC_SDA
set_property PACKAGE_PIN L20 [get_ports TX_DDC_OUT_sda_io]
set_property IOSTANDARD LVCMOS33 [get_ports TX_DDC_OUT_sda_io]

# I2C
#HDMI_CTL_SCL
set_property IOSTANDARD LVCMOS33 [get_ports HDMI_CTL_IIC_scl_io]
set_property PACKAGE_PIN K17 [get_ports HDMI_CTL_IIC_scl_io]
#HDMI_CTL_SDA
set_property IOSTANDARD LVCMOS33 [get_ports HDMI_CTL_IIC_sda_io]
set_property PACKAGE_PIN L18 [get_ports HDMI_CTL_IIC_sda_io]

#HDMI_8T49N241_LOL
set_property PACKAGE_PIN G20 [get_ports IDT_8T49N241_LOL_IN]
set_property IOSTANDARD LVCMOS33 [get_ports IDT_8T49N241_LOL_IN]
#HDMI_TX_EN
set_property PACKAGE_PIN K20 [get_ports TX_EN_OUT]
set_property IOSTANDARD LVCMOS33 [get_ports TX_EN_OUT]

#HDMI AUDIO
set_property PACKAGE_PIN AE42 [get_ports CLK_IN_AUDIO_clk_p[0]]       
set_property IOSTANDARD DIFF_LVSTL_11 [get_ports CLK_IN_AUDIO_clk_p[0]]
