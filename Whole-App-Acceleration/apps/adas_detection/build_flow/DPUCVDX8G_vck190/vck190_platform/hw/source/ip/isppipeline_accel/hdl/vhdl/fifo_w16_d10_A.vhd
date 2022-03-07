-- ==============================================================
-- Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_unsigned.all;

entity fifo_w16_d10_A_shiftReg is
    generic (
        DATA_WIDTH : integer := 16;
        ADDR_WIDTH : integer := 4;
        DEPTH : integer := 10);
    port (
        clk : in std_logic;
        data : in std_logic_vector(DATA_WIDTH-1 downto 0);
        ce : in std_logic;
        a : in std_logic_vector(ADDR_WIDTH-1 downto 0);
        q : out std_logic_vector(DATA_WIDTH-1 downto 0));
end fifo_w16_d10_A_shiftReg;

architecture rtl of fifo_w16_d10_A_shiftReg is
--constant DEPTH_WIDTH: integer := 16;
type SRL_ARRAY is array (0 to DEPTH-1) of std_logic_vector(DATA_WIDTH-1 downto 0);
signal SRL_SIG : SRL_ARRAY;

begin
p_shift: process (clk)
begin
    if (clk'event and clk = '1') then
        if (ce = '1') then
            SRL_SIG <= data & SRL_SIG(0 to DEPTH-2);
        end if;
    end if;
end process;

q <= SRL_SIG(conv_integer(a));

end rtl;

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;

entity fifo_w16_d10_A is 
    generic (
        MEM_STYLE  : string := "shiftreg"; 
        DATA_WIDTH : integer := 16;
        ADDR_WIDTH : integer := 4;
        DEPTH : integer := 10);
    port (
        clk : IN STD_LOGIC;
        reset : IN STD_LOGIC;
        if_empty_n : OUT STD_LOGIC;
        if_read_ce : IN STD_LOGIC;
        if_read : IN STD_LOGIC;
        if_dout : OUT STD_LOGIC_VECTOR(DATA_WIDTH - 1 downto 0);
        if_full_n : OUT STD_LOGIC;
        if_write_ce : IN STD_LOGIC;
        if_write : IN STD_LOGIC;
        if_din : IN STD_LOGIC_VECTOR(DATA_WIDTH - 1 downto 0));
end entity;

architecture rtl of fifo_w16_d10_A is

    component fifo_w16_d10_A_shiftReg is
    generic (
        DATA_WIDTH : integer := 16;
        ADDR_WIDTH : integer := 4;
        DEPTH : integer := 10);
    port (
        clk : in std_logic;
        data : in std_logic_vector(DATA_WIDTH-1 downto 0);
        ce : in std_logic;
        a : in std_logic_vector(ADDR_WIDTH-1 downto 0);
        q : out std_logic_vector(DATA_WIDTH-1 downto 0));
    end component;

    signal shiftReg_addr : STD_LOGIC_VECTOR(ADDR_WIDTH - 1 downto 0);
    signal shiftReg_data, shiftReg_q : STD_LOGIC_VECTOR(DATA_WIDTH - 1 downto 0);
    signal shiftReg_ce : STD_LOGIC;
    signal mOutPtr : STD_LOGIC_VECTOR(ADDR_WIDTH downto 0) := (others => '1');
    signal internal_empty_n : STD_LOGIC := '0';
    signal internal_full_n  : STD_LOGIC := '1';

begin
    if_empty_n <= internal_empty_n;
    if_full_n <= internal_full_n;
    shiftReg_data <= if_din;
    if_dout <= shiftReg_q;

    process (clk)
    begin
        if clk'event and clk = '1' then
            if reset = '1' then
                mOutPtr <= (others => '1');
                internal_empty_n <= '0';
                internal_full_n <= '1';
            else
                if ((if_read and if_read_ce) = '1' and internal_empty_n = '1') and 
                   ((if_write and if_write_ce) = '0' or internal_full_n = '0') then
                    mOutPtr <= mOutPtr - conv_std_logic_vector(1, 5);
                    if (mOutPtr = conv_std_logic_vector(0, 5)) then 
                        internal_empty_n <= '0';
                    end if;
                    internal_full_n <= '1';
                elsif ((if_read and if_read_ce) = '0' or internal_empty_n = '0') and 
                   ((if_write and if_write_ce) = '1' and internal_full_n = '1') then
                    mOutPtr <= mOutPtr + conv_std_logic_vector(1, 5);
                    internal_empty_n <= '1';
                    if (mOutPtr = conv_std_logic_vector(DEPTH, 5) - conv_std_logic_vector(2, 5)) then 
                        internal_full_n <= '0';
                    end if;
                end if;
            end if;
        end if;
    end process;

    shiftReg_addr <= (others => '0') when mOutPtr(ADDR_WIDTH) = '1' else mOutPtr(ADDR_WIDTH-1 downto 0);
    shiftReg_ce <= (if_write and if_write_ce) and internal_full_n;

    U_fifo_w16_d10_A_shiftReg : fifo_w16_d10_A_shiftReg
    generic map (
        DATA_WIDTH => DATA_WIDTH,
        ADDR_WIDTH => ADDR_WIDTH,
        DEPTH => DEPTH)
    port map (
        clk => clk,
        data => shiftReg_data,
        ce => shiftReg_ce,
        a => shiftReg_addr,
        q => shiftReg_q);

end rtl;

