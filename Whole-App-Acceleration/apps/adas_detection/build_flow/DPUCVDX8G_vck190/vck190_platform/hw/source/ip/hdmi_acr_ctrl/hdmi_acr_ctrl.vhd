
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
USE ieee.std_logic_arith.all;
USE ieee.std_logic_unsigned.all;

entity hdmi_acr_ctrl is

  port
  (
  -- AXI4-Lite bus (cpu control)
  axi_aclk    : in std_logic;
  axi_aresetn : in std_logic;
  -- - Write address
  axi_awvalid : in std_logic;
  axi_awready : out std_logic;
  axi_awaddr  : in std_logic_vector(31 downto 0);
  -- - Write data
  axi_wvalid : in std_logic;
  axi_wready : out std_logic;
  axi_wdata  : in std_logic_vector(31 downto 0);
  axi_wstrb  : in std_logic_vector(3 downto 0);
  -- - Write response
  axi_bvalid : out std_logic;
  axi_bready : in std_logic;
  axi_bresp  : out std_logic_vector(1 downto 0);
  -- - Read address  
  axi_arvalid : in std_logic;
  axi_arready : out std_logic;
  axi_araddr  : in std_logic_vector(31 downto 0);
  -- - Read data/response
  axi_rvalid : out std_logic;
  axi_rready : in std_logic;
  axi_rdata  : out std_logic_vector(31 downto 0);
  axi_rresp  : out std_logic_vector(1 downto 0);
   
  -- Audio clock (512 * Fs)
  aud_clk : in std_logic;
   
  -- HDMI TMDS clock 
  hdmi_clk : in std_logic;

  -- Audio PLL Lock
  pll_lock_in : in std_logic;
  
  -- Audio reset out
  aud_resetn_out : out std_logic;
  
  -- Audio Clock Regeneration values
  aud_acr_valid_in : in std_logic;
  aud_acr_cts_in : in std_logic_vector(19 downto 0);
  aud_acr_n_in : in std_logic_vector(19 downto 0);
  aud_acr_valid_out: out std_logic;
  aud_acr_cts_out : out std_logic_vector(19 downto 0);
  aud_acr_n_out : out std_logic_vector(19 downto 0)
  );
  
end hdmi_acr_ctrl;

architecture rtl of hdmi_acr_ctrl is

  ATTRIBUTE X_INTERFACE_INFO : STRING;
  ATTRIBUTE X_INTERFACE_PARAMETER : STRING;
  
  ATTRIBUTE X_INTERFACE_INFO OF axi_aclk: SIGNAL IS "xilinx.com:signal:clock:1.0 axi_aclk CLK";
  ATTRIBUTE X_INTERFACE_PARAMETER OF axi_aclk: SIGNAL IS "ASSOCIATED_BUSIF axi, ASSOCIATED_RESET axi_aresetn";
  ATTRIBUTE X_INTERFACE_INFO OF axi_aresetn: SIGNAL IS "xilinx.com:signal:reset:1.0 axi_aresetn RST";
  ATTRIBUTE X_INTERFACE_PARAMETER OF axi_aresetn: SIGNAL IS "POLARITY ACTIVE_LOW";  
  
component hdmi_acr_ctrl_top 
  port
  (
  -- AXI4-Lite bus (cpu control)
  axi_aclk    : in std_logic;
  axi_aresetn : in std_logic;
  -- - Write address
  axi_awvalid : in std_logic;
  axi_awready : out std_logic;
  axi_awaddr  : in std_logic_vector(31 downto 0);
  -- - Write data
  axi_wvalid : in std_logic;
  axi_wready : out std_logic;
  axi_wdata  : in std_logic_vector(31 downto 0);
  axi_wstrb  : in std_logic_vector(3 downto 0);
  -- - Write response
  axi_bvalid : out std_logic;
  axi_bready : in std_logic;
  axi_bresp  : out std_logic_vector(1 downto 0);
  -- - Read address  
  axi_arvalid : in std_logic;
  axi_arready : out std_logic;
  axi_araddr  : in std_logic_vector(31 downto 0);
  -- - Read data/response
  axi_rvalid : out std_logic;
  axi_rready : in std_logic;
  axi_rdata  : out std_logic_vector(31 downto 0);
  axi_rresp  : out std_logic_vector(1 downto 0);
   
  -- Audio clock (512 * Fs)
  aud_clk : in std_logic;
   
  -- HDMI TMDS clock 
  hdmi_clk : in std_logic;

  -- Audio PLL Lock
  pll_lock_in : in std_logic;
  
  -- Audio reset out
  aud_resetn_out : out std_logic;
  
  -- Audio Clock Regeneration values
  aud_acr_valid_in : in std_logic;
  aud_acr_cts_in : in std_logic_vector(19 downto 0);
  aud_acr_n_in : in std_logic_vector(19 downto 0);
  aud_acr_valid_out: out std_logic;
  aud_acr_cts_out : out std_logic_vector(19 downto 0);
  aud_acr_n_out : out std_logic_vector(19 downto 0)
  );
  end component;

begin

hdmi_acr_ctrl_top_inst : hdmi_acr_ctrl_top
port map (
   -- AXI4-Lite bus (cpu control)
   axi_aclk               =>     axi_aclk   , 
   axi_aresetn            =>     axi_aresetn, 
   -- Write address                         
   axi_awvalid            =>     axi_awvalid,
   axi_awready            =>     axi_awready,
   axi_awaddr             =>     axi_awaddr ,
   -- Write data                            
   axi_wvalid             =>     axi_wvalid ,
   axi_wready             =>     axi_wready ,
   axi_wdata              =>     axi_wdata  ,
   axi_wstrb              =>     axi_wstrb  ,
   -- Write response                        
   axi_bvalid             =>     axi_bvalid ,
   axi_bready             =>     axi_bready ,
   axi_bresp              =>     axi_bresp  ,
   -- Read address                          
   axi_arvalid            =>     axi_arvalid,
   axi_arready            =>     axi_arready,
   axi_araddr             =>     axi_araddr ,
   -- Read data/response                    
   axi_rvalid             =>     axi_rvalid ,
   axi_rready             =>     axi_rready ,
   axi_rdata              =>     axi_rdata  ,
   axi_rresp              =>     axi_rresp  ,

  -- Audio clock (512 * Fs)
  aud_clk                 =>     aud_clk,
   
  -- HDMI TMDS clock 
  hdmi_clk                =>     hdmi_clk,

  -- Audio PLL Lock
  pll_lock_in             =>     pll_lock_in, 
  
  -- Audio reset out
  aud_resetn_out          =>     aud_resetn_out,
  
  -- Audio Clock Regeneration values
  aud_acr_valid_in        => aud_acr_valid_in, 
  aud_acr_cts_in          => aud_acr_cts_in, 
  aud_acr_n_in            => aud_acr_n_in, 
  aud_acr_valid_out       => aud_acr_valid_out,
  aud_acr_cts_out         => aud_acr_cts_out, 
  aud_acr_n_out           => aud_acr_n_out 
);

end rtl;

