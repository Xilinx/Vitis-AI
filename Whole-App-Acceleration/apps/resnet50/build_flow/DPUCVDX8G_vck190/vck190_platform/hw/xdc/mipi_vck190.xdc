#
# MIPI CSI-2
#
# PL Port                    Pin   Schematic        FMC
#
# csi_mipi_phy_if_clk_p      BD23  FMCP1_LA00_CC_P   G6
# csi_mipi_phy_if_data_p[0]  BG25  FMCP1_LA10_P     C14
# csi_mipi_phy_if_data_p[1]  BC23  FMCP1_LA01_CC_P   D8
# csi_mipi_phy_if_data_p[2]  BC22  FMCP1_LA08_P     G12
# csi_mipi_phy_if_data_p[3]  BC25  FMCP1_LA07_P     H12
#
set_property PACKAGE_PIN BD23 [get_ports csi_mipi_phy_if_clk_p]
set_property PACKAGE_PIN BG25 [get_ports {csi_mipi_phy_if_data_p[0]}]
set_property PACKAGE_PIN BC23 [get_ports {csi_mipi_phy_if_data_p[1]}]
set_property PACKAGE_PIN BC22 [get_ports {csi_mipi_phy_if_data_p[2]}]
set_property PACKAGE_PIN BC25 [get_ports {csi_mipi_phy_if_data_p[3]}]
set_property IOSTANDARD MIPI_DPHY [get_ports csi_mipi_phy_if_*]
#
# MIPI - I2C IMX274 Sensor
#
# PL Port            Pin   Schematic     FMC
#
# sensor_iic_scl_io  BC18  FMCP1_LA26_P  D26  FMC_SCL
# sensor_iic_sda_io  BD18  FMCP1_LA26_N  D27  FMC_SDA
#
set_property PACKAGE_PIN BC18 [get_ports sensor_iic_scl_io]
set_property PACKAGE_PIN BD18 [get_ports sensor_iic_sda_io]
set_property PULLUP true [get_ports sensor_iic_*]
set_property DRIVE 8 [get_ports sensor_iic_*]
set_property IOSTANDARD LVCMOS12 [get_ports sensor_iic_*]
#
# MIPI - GPIO - CAM_FLASH, CAM_XCE, CAM_RST
#
# PL Port               Pin   Schematic     FMC
#
# sensor_gpio_rst       BG18  FMCP1_LA22_N  G25  FMC_RST
# sensor_gpio_spi_cs_n  BF19  FMCP1_LA27_P  C26  FMC_SPI_CS_N
# sensor_gpio_flash     BG20  FMCP1_LA16_N  G19  FMC_FLASH
#
set_property PACKAGE_PIN BG18 [get_ports {sensor_gpio_rst}]
set_property PACKAGE_PIN BF19 [get_ports {sensor_gpio_spi_cs_n}]
set_property PACKAGE_PIN BG20 [get_ports {sensor_gpio_flash}]
set_property IOSTANDARD LVCMOS12 [get_ports sensor_gpio_*]
set_property DIFF_TERM_ADV TERM_100 [get_ports csi_mipi_phy_if_clk*]
set_property DIFF_TERM_ADV TERM_100 [get_ports {csi_mipi_phy_if_data_p[*]}]
set_property DIFF_TERM_ADV TERM_100 [get_ports {csi_mipi_phy_if_data_n[*]}]
