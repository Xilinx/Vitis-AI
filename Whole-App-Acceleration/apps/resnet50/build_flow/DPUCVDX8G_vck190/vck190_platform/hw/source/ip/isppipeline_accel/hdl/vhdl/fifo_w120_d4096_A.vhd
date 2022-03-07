-- ==============================================================
-- Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity fifo_w120_d4096_A is
    generic (
        MEM_STYLE   : string  := "block";
        DATA_WIDTH  : natural := 120;
        ADDR_WIDTH  : natural := 12;
        DEPTH       : natural := 4096
    );
    port (
        clk         : in  std_logic;
        reset       : in  std_logic;
        if_full_n   : out std_logic;
        if_write_ce : in  std_logic;
        if_write    : in  std_logic;
        if_din      : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        if_empty_n  : out std_logic;
        if_read_ce  : in  std_logic;
        if_read     : in  std_logic;
        if_dout     : out std_logic_vector(DATA_WIDTH - 1 downto 0)
    );
end entity;

architecture arch of fifo_w120_d4096_A is
    type memtype is array (0 to DEPTH - 1) of std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal mem        : memtype;
    signal q_buf      : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal waddr      : unsigned(ADDR_WIDTH - 1 downto 0) := (others => '0');
    signal raddr      : unsigned(ADDR_WIDTH - 1 downto 0) := (others => '0');
    signal wnext      : unsigned(ADDR_WIDTH - 1 downto 0);
    signal rnext      : unsigned(ADDR_WIDTH - 1 downto 0);
    signal push       : std_logic;
    signal pop        : std_logic;
    signal usedw      : unsigned(ADDR_WIDTH - 1 downto 0) := (others => '0');
    signal full_n     : std_logic := '1';
    signal empty_n    : std_logic := '0';
    signal q_tmp      : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal show_ahead : std_logic := '0';
    signal dout_buf   : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal dout_valid : std_logic := '0';
    attribute ram_style: string;
    attribute ram_style of mem: signal is MEM_STYLE;
begin
    if_full_n  <= full_n;
    if_empty_n <= dout_valid;
    if_dout    <= dout_buf;
    push       <= full_n and if_write_ce and if_write;
    pop        <= empty_n and if_read_ce and (not dout_valid or if_read);
    wnext      <= waddr when push = '0' else
                  (others => '0') when waddr = DEPTH - 1 else
                  waddr + 1;
    rnext      <= raddr when pop = '0' else
                  (others => '0') when raddr = DEPTH - 1 else
                  raddr + 1;

    -- waddr
    process (clk) begin
        if clk'event and clk = '1' then
            if reset = '1' then
                waddr <= (others => '0');
            else
                waddr <= wnext;
            end if;
        end if;
    end process;

    -- raddr
    process (clk) begin
        if clk'event and clk = '1' then
            if reset = '1' then
                raddr <= (others => '0');
            else
                raddr <= rnext;
            end if;
        end if;
    end process;

    -- usedw
    process (clk) begin
        if clk'event and clk = '1' then
            if reset = '1' then
                usedw <= (others => '0');
            elsif push = '1' and pop = '0' then
                usedw <= usedw + 1;
            elsif push = '0' and pop = '1' then
                usedw <= usedw - 1;
            end if;
        end if;
    end process;

    -- full_n
    process (clk) begin
        if clk'event and clk = '1' then
            if reset = '1' then
                full_n <= '1';
            elsif push = '1' and pop = '0' then
                if usedw = DEPTH - 1 then
                    full_n <= '0';
                else
                    full_n <= '1';
                end if;
            elsif push = '0' and pop = '1' then
                full_n <= '1';
            end if;
        end if;
    end process;

    -- empty_n
    process (clk) begin
        if clk'event and clk = '1' then
            if reset = '1' then
                empty_n <= '0';
            elsif push = '1' and pop = '0' then
                empty_n <= '1';
            elsif push = '0' and pop = '1' then
                if usedw = 1 then
                    empty_n <= '0';
                else
                    empty_n <= '1';
                end if;
            end if;
        end if;
    end process;

    -- mem
    process (clk) begin
        if clk'event and clk = '1' then
            if push = '1' then
                mem(to_integer(waddr)) <= if_din;
            end if;
        end if;
    end process;

    -- q_buf
    process (clk) begin
        if clk'event and clk = '1' then
            q_buf <= mem(to_integer(rnext));
        end if;
    end process;

    -- q_tmp
    process (clk) begin
        if clk'event and clk = '1' then
            if reset = '1' then
                q_tmp <= (others => '0');
            elsif push = '1' then
                q_tmp <= if_din;
            end if;
        end if;
    end process;

    -- show_ahead
    process (clk) begin
        if clk'event and clk = '1' then
            if reset = '1' then
                show_ahead <= '0';
            elsif push = '1' and (usedw = 0 or (usedw = 1 and pop = '1')) then
                show_ahead <= '1';
            else
                show_ahead <= '0';
            end if;
        end if;
    end process;

    -- dout_buf
    process (clk) begin
        if clk'event and clk = '1' then
            if reset = '1' then
                dout_buf <= (others => '0');
            elsif pop = '1' then
                if show_ahead = '1' then
                    dout_buf <= q_tmp;
                else
                    dout_buf <= q_buf;
                end if;
            end if;
        end if;
    end process;

    -- dout_valid
    process (clk) begin
        if clk'event and clk = '1' then
            if reset = '1' then
                dout_valid <= '0';
            elsif pop = '1' then
                dout_valid <= '1';
            elsif if_read_ce = '1' and if_read = '1' then
                dout_valid <= '0';
            end if;
        end if;
    end process;
end architecture;

