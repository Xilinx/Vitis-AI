-- ==============================================================
-- Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.NUMERIC_STD.all;

entity ISPPipeline_acceldEe is
generic (
    ID            :integer := 0;
    NUM_STAGE     :integer := 1;
    din0_WIDTH       :integer := 32;
    din1_WIDTH       :integer := 32;
    din2_WIDTH       :integer := 32;
    din3_WIDTH       :integer := 32;
    din4_WIDTH       :integer := 32;
    din5_WIDTH       :integer := 32;
    dout_WIDTH        :integer := 32);
port (
    din0   :in  std_logic_vector(39 downto 0);
    din1   :in  std_logic_vector(39 downto 0);
    din2   :in  std_logic_vector(39 downto 0);
    din3   :in  std_logic_vector(39 downto 0);
    din4   :in  std_logic_vector(39 downto 0);
    din5   :in  std_logic_vector(2 downto 0);
    dout     :out std_logic_vector(39 downto 0));
end entity;

architecture rtl of ISPPipeline_acceldEe is
    -- puts internal signals
    signal sel    : std_logic_vector(2 downto 0);
    -- level 1 signals
    signal mux_1_0    : std_logic_vector(39 downto 0);
    signal mux_1_1    : std_logic_vector(39 downto 0);
    signal mux_1_2    : std_logic_vector(39 downto 0);
    -- level 2 signals
    signal mux_2_0    : std_logic_vector(39 downto 0);
    signal mux_2_1    : std_logic_vector(39 downto 0);
    -- level 3 signals
    signal mux_3_0    : std_logic_vector(39 downto 0);
begin

sel <= din5;

-- Generate level 1 logic
mux_1_0 <= din0 when sel(0) = '0' else din1;
mux_1_1 <= din2 when sel(0) = '0' else din3;
mux_1_2 <= din4;

-- Generate level 2 logic
mux_2_0 <= mux_1_0 when sel(1) = '0' else mux_1_1;
mux_2_1 <= mux_1_2;

-- Generate level 3 logic
mux_3_0 <= mux_2_0 when sel(2) = '0' else mux_2_1;

-- output logic
dout <= mux_3_0;

end architecture;
