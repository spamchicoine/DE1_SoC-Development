	component Computer_System is
		port (
			adc_sclk                        : out   std_logic;                                        -- sclk
			adc_cs_n                        : out   std_logic;                                        -- cs_n
			adc_dout                        : in    std_logic                     := 'X';             -- dout
			adc_din                         : out   std_logic;                                        -- din
			audio_ADCDAT                    : in    std_logic                     := 'X';             -- ADCDAT
			audio_ADCLRCK                   : in    std_logic                     := 'X';             -- ADCLRCK
			audio_BCLK                      : in    std_logic                     := 'X';             -- BCLK
			audio_DACDAT                    : out   std_logic;                                        -- DACDAT
			audio_DACLRCK                   : in    std_logic                     := 'X';             -- DACLRCK
			audio_clk_clk                   : out   std_logic;                                        -- clk
			audio_pll_ref_clk_clk           : in    std_logic                     := 'X';             -- clk
			audio_pll_ref_reset_reset       : in    std_logic                     := 'X';             -- reset
			av_config_SDAT                  : inout std_logic                     := 'X';             -- SDAT
			av_config_SCLK                  : out   std_logic;                                        -- SCLK
			d5m_config_I2C_SDAT             : inout std_logic                     := 'X';             -- I2C_SDAT
			d5m_config_I2C_SCLK             : out   std_logic;                                        -- I2C_SCLK
			d5m_config_exposure             : in    std_logic_vector(15 downto 0) := (others => 'X'); -- exposure
			expansion_jp1_export            : inout std_logic_vector(31 downto 0) := (others => 'X'); -- export
			hex3_hex0_export                : out   std_logic_vector(31 downto 0);                    -- export
			hex5_hex4_export                : out   std_logic_vector(15 downto 0);                    -- export
			hps_io_hps_io_emac1_inst_TX_CLK : out   std_logic;                                        -- hps_io_emac1_inst_TX_CLK
			hps_io_hps_io_emac1_inst_TXD0   : out   std_logic;                                        -- hps_io_emac1_inst_TXD0
			hps_io_hps_io_emac1_inst_TXD1   : out   std_logic;                                        -- hps_io_emac1_inst_TXD1
			hps_io_hps_io_emac1_inst_TXD2   : out   std_logic;                                        -- hps_io_emac1_inst_TXD2
			hps_io_hps_io_emac1_inst_TXD3   : out   std_logic;                                        -- hps_io_emac1_inst_TXD3
			hps_io_hps_io_emac1_inst_RXD0   : in    std_logic                     := 'X';             -- hps_io_emac1_inst_RXD0
			hps_io_hps_io_emac1_inst_MDIO   : inout std_logic                     := 'X';             -- hps_io_emac1_inst_MDIO
			hps_io_hps_io_emac1_inst_MDC    : out   std_logic;                                        -- hps_io_emac1_inst_MDC
			hps_io_hps_io_emac1_inst_RX_CTL : in    std_logic                     := 'X';             -- hps_io_emac1_inst_RX_CTL
			hps_io_hps_io_emac1_inst_TX_CTL : out   std_logic;                                        -- hps_io_emac1_inst_TX_CTL
			hps_io_hps_io_emac1_inst_RX_CLK : in    std_logic                     := 'X';             -- hps_io_emac1_inst_RX_CLK
			hps_io_hps_io_emac1_inst_RXD1   : in    std_logic                     := 'X';             -- hps_io_emac1_inst_RXD1
			hps_io_hps_io_emac1_inst_RXD2   : in    std_logic                     := 'X';             -- hps_io_emac1_inst_RXD2
			hps_io_hps_io_emac1_inst_RXD3   : in    std_logic                     := 'X';             -- hps_io_emac1_inst_RXD3
			hps_io_hps_io_qspi_inst_IO0     : inout std_logic                     := 'X';             -- hps_io_qspi_inst_IO0
			hps_io_hps_io_qspi_inst_IO1     : inout std_logic                     := 'X';             -- hps_io_qspi_inst_IO1
			hps_io_hps_io_qspi_inst_IO2     : inout std_logic                     := 'X';             -- hps_io_qspi_inst_IO2
			hps_io_hps_io_qspi_inst_IO3     : inout std_logic                     := 'X';             -- hps_io_qspi_inst_IO3
			hps_io_hps_io_qspi_inst_SS0     : out   std_logic;                                        -- hps_io_qspi_inst_SS0
			hps_io_hps_io_qspi_inst_CLK     : out   std_logic;                                        -- hps_io_qspi_inst_CLK
			hps_io_hps_io_sdio_inst_CMD     : inout std_logic                     := 'X';             -- hps_io_sdio_inst_CMD
			hps_io_hps_io_sdio_inst_D0      : inout std_logic                     := 'X';             -- hps_io_sdio_inst_D0
			hps_io_hps_io_sdio_inst_D1      : inout std_logic                     := 'X';             -- hps_io_sdio_inst_D1
			hps_io_hps_io_sdio_inst_CLK     : out   std_logic;                                        -- hps_io_sdio_inst_CLK
			hps_io_hps_io_sdio_inst_D2      : inout std_logic                     := 'X';             -- hps_io_sdio_inst_D2
			hps_io_hps_io_sdio_inst_D3      : inout std_logic                     := 'X';             -- hps_io_sdio_inst_D3
			hps_io_hps_io_usb1_inst_D0      : inout std_logic                     := 'X';             -- hps_io_usb1_inst_D0
			hps_io_hps_io_usb1_inst_D1      : inout std_logic                     := 'X';             -- hps_io_usb1_inst_D1
			hps_io_hps_io_usb1_inst_D2      : inout std_logic                     := 'X';             -- hps_io_usb1_inst_D2
			hps_io_hps_io_usb1_inst_D3      : inout std_logic                     := 'X';             -- hps_io_usb1_inst_D3
			hps_io_hps_io_usb1_inst_D4      : inout std_logic                     := 'X';             -- hps_io_usb1_inst_D4
			hps_io_hps_io_usb1_inst_D5      : inout std_logic                     := 'X';             -- hps_io_usb1_inst_D5
			hps_io_hps_io_usb1_inst_D6      : inout std_logic                     := 'X';             -- hps_io_usb1_inst_D6
			hps_io_hps_io_usb1_inst_D7      : inout std_logic                     := 'X';             -- hps_io_usb1_inst_D7
			hps_io_hps_io_usb1_inst_CLK     : in    std_logic                     := 'X';             -- hps_io_usb1_inst_CLK
			hps_io_hps_io_usb1_inst_STP     : out   std_logic;                                        -- hps_io_usb1_inst_STP
			hps_io_hps_io_usb1_inst_DIR     : in    std_logic                     := 'X';             -- hps_io_usb1_inst_DIR
			hps_io_hps_io_usb1_inst_NXT     : in    std_logic                     := 'X';             -- hps_io_usb1_inst_NXT
			hps_io_hps_io_spim1_inst_CLK    : out   std_logic;                                        -- hps_io_spim1_inst_CLK
			hps_io_hps_io_spim1_inst_MOSI   : out   std_logic;                                        -- hps_io_spim1_inst_MOSI
			hps_io_hps_io_spim1_inst_MISO   : in    std_logic                     := 'X';             -- hps_io_spim1_inst_MISO
			hps_io_hps_io_spim1_inst_SS0    : out   std_logic;                                        -- hps_io_spim1_inst_SS0
			hps_io_hps_io_uart0_inst_RX     : in    std_logic                     := 'X';             -- hps_io_uart0_inst_RX
			hps_io_hps_io_uart0_inst_TX     : out   std_logic;                                        -- hps_io_uart0_inst_TX
			hps_io_hps_io_i2c0_inst_SDA     : inout std_logic                     := 'X';             -- hps_io_i2c0_inst_SDA
			hps_io_hps_io_i2c0_inst_SCL     : inout std_logic                     := 'X';             -- hps_io_i2c0_inst_SCL
			hps_io_hps_io_i2c1_inst_SDA     : inout std_logic                     := 'X';             -- hps_io_i2c1_inst_SDA
			hps_io_hps_io_i2c1_inst_SCL     : inout std_logic                     := 'X';             -- hps_io_i2c1_inst_SCL
			hps_io_hps_io_gpio_inst_GPIO09  : inout std_logic                     := 'X';             -- hps_io_gpio_inst_GPIO09
			hps_io_hps_io_gpio_inst_GPIO35  : inout std_logic                     := 'X';             -- hps_io_gpio_inst_GPIO35
			hps_io_hps_io_gpio_inst_GPIO40  : inout std_logic                     := 'X';             -- hps_io_gpio_inst_GPIO40
			hps_io_hps_io_gpio_inst_GPIO41  : inout std_logic                     := 'X';             -- hps_io_gpio_inst_GPIO41
			hps_io_hps_io_gpio_inst_GPIO48  : inout std_logic                     := 'X';             -- hps_io_gpio_inst_GPIO48
			hps_io_hps_io_gpio_inst_GPIO53  : inout std_logic                     := 'X';             -- hps_io_gpio_inst_GPIO53
			hps_io_hps_io_gpio_inst_GPIO54  : inout std_logic                     := 'X';             -- hps_io_gpio_inst_GPIO54
			hps_io_hps_io_gpio_inst_GPIO61  : inout std_logic                     := 'X';             -- hps_io_gpio_inst_GPIO61
			irda_TXD                        : out   std_logic;                                        -- TXD
			irda_RXD                        : in    std_logic                     := 'X';             -- RXD
			leds_export                     : out   std_logic_vector(9 downto 0);                     -- export
			memory_mem_a                    : out   std_logic_vector(14 downto 0);                    -- mem_a
			memory_mem_ba                   : out   std_logic_vector(2 downto 0);                     -- mem_ba
			memory_mem_ck                   : out   std_logic;                                        -- mem_ck
			memory_mem_ck_n                 : out   std_logic;                                        -- mem_ck_n
			memory_mem_cke                  : out   std_logic;                                        -- mem_cke
			memory_mem_cs_n                 : out   std_logic;                                        -- mem_cs_n
			memory_mem_ras_n                : out   std_logic;                                        -- mem_ras_n
			memory_mem_cas_n                : out   std_logic;                                        -- mem_cas_n
			memory_mem_we_n                 : out   std_logic;                                        -- mem_we_n
			memory_mem_reset_n              : out   std_logic;                                        -- mem_reset_n
			memory_mem_dq                   : inout std_logic_vector(31 downto 0) := (others => 'X'); -- mem_dq
			memory_mem_dqs                  : inout std_logic_vector(3 downto 0)  := (others => 'X'); -- mem_dqs
			memory_mem_dqs_n                : inout std_logic_vector(3 downto 0)  := (others => 'X'); -- mem_dqs_n
			memory_mem_odt                  : out   std_logic;                                        -- mem_odt
			memory_mem_dm                   : out   std_logic_vector(3 downto 0);                     -- mem_dm
			memory_oct_rzqin                : in    std_logic                     := 'X';             -- oct_rzqin
			ps2_port_CLK                    : inout std_logic                     := 'X';             -- CLK
			ps2_port_DAT                    : inout std_logic                     := 'X';             -- DAT
			ps2_port_dual_CLK               : inout std_logic                     := 'X';             -- CLK
			ps2_port_dual_DAT               : inout std_logic                     := 'X';             -- DAT
			pushbuttons_export              : in    std_logic_vector(3 downto 0)  := (others => 'X'); -- export
			sdram_addr                      : out   std_logic_vector(12 downto 0);                    -- addr
			sdram_ba                        : out   std_logic_vector(1 downto 0);                     -- ba
			sdram_cas_n                     : out   std_logic;                                        -- cas_n
			sdram_cke                       : out   std_logic;                                        -- cke
			sdram_cs_n                      : out   std_logic;                                        -- cs_n
			sdram_dq                        : inout std_logic_vector(15 downto 0) := (others => 'X'); -- dq
			sdram_dqm                       : out   std_logic_vector(1 downto 0);                     -- dqm
			sdram_ras_n                     : out   std_logic;                                        -- ras_n
			sdram_we_n                      : out   std_logic;                                        -- we_n
			sdram_clk_clk                   : out   std_logic;                                        -- clk
			slider_switches_export          : in    std_logic_vector(9 downto 0)  := (others => 'X'); -- export
			system_pll_ref_clk_clk          : in    std_logic                     := 'X';             -- clk
			system_pll_ref_reset_reset      : in    std_logic                     := 'X';             -- reset
			vga_CLK                         : out   std_logic;                                        -- CLK
			vga_HS                          : out   std_logic;                                        -- HS
			vga_VS                          : out   std_logic;                                        -- VS
			vga_BLANK                       : out   std_logic;                                        -- BLANK
			vga_SYNC                        : out   std_logic;                                        -- SYNC
			vga_R                           : out   std_logic_vector(7 downto 0);                     -- R
			vga_G                           : out   std_logic_vector(7 downto 0);                     -- G
			vga_B                           : out   std_logic_vector(7 downto 0);                     -- B
			vga_pll_d5m_clk_clk             : out   std_logic;                                        -- clk
			vga_pll_ref_clk_clk             : in    std_logic                     := 'X';             -- clk
			vga_pll_ref_reset_reset         : in    std_logic                     := 'X';             -- reset
			video_in_PIXEL_CLK              : in    std_logic                     := 'X';             -- PIXEL_CLK
			video_in_LINE_VALID             : in    std_logic                     := 'X';             -- LINE_VALID
			video_in_FRAME_VALID            : in    std_logic                     := 'X';             -- FRAME_VALID
			video_in_pixel_clk_reset        : in    std_logic                     := 'X';             -- pixel_clk_reset
			video_in_PIXEL_DATA             : in    std_logic_vector(11 downto 0) := (others => 'X')  -- PIXEL_DATA
		);
	end component Computer_System;

	u0 : component Computer_System
		port map (
			adc_sclk                        => CONNECTED_TO_adc_sclk,                        --                  adc.sclk
			adc_cs_n                        => CONNECTED_TO_adc_cs_n,                        --                     .cs_n
			adc_dout                        => CONNECTED_TO_adc_dout,                        --                     .dout
			adc_din                         => CONNECTED_TO_adc_din,                         --                     .din
			audio_ADCDAT                    => CONNECTED_TO_audio_ADCDAT,                    --                audio.ADCDAT
			audio_ADCLRCK                   => CONNECTED_TO_audio_ADCLRCK,                   --                     .ADCLRCK
			audio_BCLK                      => CONNECTED_TO_audio_BCLK,                      --                     .BCLK
			audio_DACDAT                    => CONNECTED_TO_audio_DACDAT,                    --                     .DACDAT
			audio_DACLRCK                   => CONNECTED_TO_audio_DACLRCK,                   --                     .DACLRCK
			audio_clk_clk                   => CONNECTED_TO_audio_clk_clk,                   --            audio_clk.clk
			audio_pll_ref_clk_clk           => CONNECTED_TO_audio_pll_ref_clk_clk,           --    audio_pll_ref_clk.clk
			audio_pll_ref_reset_reset       => CONNECTED_TO_audio_pll_ref_reset_reset,       --  audio_pll_ref_reset.reset
			av_config_SDAT                  => CONNECTED_TO_av_config_SDAT,                  --            av_config.SDAT
			av_config_SCLK                  => CONNECTED_TO_av_config_SCLK,                  --                     .SCLK
			d5m_config_I2C_SDAT             => CONNECTED_TO_d5m_config_I2C_SDAT,             --           d5m_config.I2C_SDAT
			d5m_config_I2C_SCLK             => CONNECTED_TO_d5m_config_I2C_SCLK,             --                     .I2C_SCLK
			d5m_config_exposure             => CONNECTED_TO_d5m_config_exposure,             --                     .exposure
			expansion_jp1_export            => CONNECTED_TO_expansion_jp1_export,            --        expansion_jp1.export
			hex3_hex0_export                => CONNECTED_TO_hex3_hex0_export,                --            hex3_hex0.export
			hex5_hex4_export                => CONNECTED_TO_hex5_hex4_export,                --            hex5_hex4.export
			hps_io_hps_io_emac1_inst_TX_CLK => CONNECTED_TO_hps_io_hps_io_emac1_inst_TX_CLK, --               hps_io.hps_io_emac1_inst_TX_CLK
			hps_io_hps_io_emac1_inst_TXD0   => CONNECTED_TO_hps_io_hps_io_emac1_inst_TXD0,   --                     .hps_io_emac1_inst_TXD0
			hps_io_hps_io_emac1_inst_TXD1   => CONNECTED_TO_hps_io_hps_io_emac1_inst_TXD1,   --                     .hps_io_emac1_inst_TXD1
			hps_io_hps_io_emac1_inst_TXD2   => CONNECTED_TO_hps_io_hps_io_emac1_inst_TXD2,   --                     .hps_io_emac1_inst_TXD2
			hps_io_hps_io_emac1_inst_TXD3   => CONNECTED_TO_hps_io_hps_io_emac1_inst_TXD3,   --                     .hps_io_emac1_inst_TXD3
			hps_io_hps_io_emac1_inst_RXD0   => CONNECTED_TO_hps_io_hps_io_emac1_inst_RXD0,   --                     .hps_io_emac1_inst_RXD0
			hps_io_hps_io_emac1_inst_MDIO   => CONNECTED_TO_hps_io_hps_io_emac1_inst_MDIO,   --                     .hps_io_emac1_inst_MDIO
			hps_io_hps_io_emac1_inst_MDC    => CONNECTED_TO_hps_io_hps_io_emac1_inst_MDC,    --                     .hps_io_emac1_inst_MDC
			hps_io_hps_io_emac1_inst_RX_CTL => CONNECTED_TO_hps_io_hps_io_emac1_inst_RX_CTL, --                     .hps_io_emac1_inst_RX_CTL
			hps_io_hps_io_emac1_inst_TX_CTL => CONNECTED_TO_hps_io_hps_io_emac1_inst_TX_CTL, --                     .hps_io_emac1_inst_TX_CTL
			hps_io_hps_io_emac1_inst_RX_CLK => CONNECTED_TO_hps_io_hps_io_emac1_inst_RX_CLK, --                     .hps_io_emac1_inst_RX_CLK
			hps_io_hps_io_emac1_inst_RXD1   => CONNECTED_TO_hps_io_hps_io_emac1_inst_RXD1,   --                     .hps_io_emac1_inst_RXD1
			hps_io_hps_io_emac1_inst_RXD2   => CONNECTED_TO_hps_io_hps_io_emac1_inst_RXD2,   --                     .hps_io_emac1_inst_RXD2
			hps_io_hps_io_emac1_inst_RXD3   => CONNECTED_TO_hps_io_hps_io_emac1_inst_RXD3,   --                     .hps_io_emac1_inst_RXD3
			hps_io_hps_io_qspi_inst_IO0     => CONNECTED_TO_hps_io_hps_io_qspi_inst_IO0,     --                     .hps_io_qspi_inst_IO0
			hps_io_hps_io_qspi_inst_IO1     => CONNECTED_TO_hps_io_hps_io_qspi_inst_IO1,     --                     .hps_io_qspi_inst_IO1
			hps_io_hps_io_qspi_inst_IO2     => CONNECTED_TO_hps_io_hps_io_qspi_inst_IO2,     --                     .hps_io_qspi_inst_IO2
			hps_io_hps_io_qspi_inst_IO3     => CONNECTED_TO_hps_io_hps_io_qspi_inst_IO3,     --                     .hps_io_qspi_inst_IO3
			hps_io_hps_io_qspi_inst_SS0     => CONNECTED_TO_hps_io_hps_io_qspi_inst_SS0,     --                     .hps_io_qspi_inst_SS0
			hps_io_hps_io_qspi_inst_CLK     => CONNECTED_TO_hps_io_hps_io_qspi_inst_CLK,     --                     .hps_io_qspi_inst_CLK
			hps_io_hps_io_sdio_inst_CMD     => CONNECTED_TO_hps_io_hps_io_sdio_inst_CMD,     --                     .hps_io_sdio_inst_CMD
			hps_io_hps_io_sdio_inst_D0      => CONNECTED_TO_hps_io_hps_io_sdio_inst_D0,      --                     .hps_io_sdio_inst_D0
			hps_io_hps_io_sdio_inst_D1      => CONNECTED_TO_hps_io_hps_io_sdio_inst_D1,      --                     .hps_io_sdio_inst_D1
			hps_io_hps_io_sdio_inst_CLK     => CONNECTED_TO_hps_io_hps_io_sdio_inst_CLK,     --                     .hps_io_sdio_inst_CLK
			hps_io_hps_io_sdio_inst_D2      => CONNECTED_TO_hps_io_hps_io_sdio_inst_D2,      --                     .hps_io_sdio_inst_D2
			hps_io_hps_io_sdio_inst_D3      => CONNECTED_TO_hps_io_hps_io_sdio_inst_D3,      --                     .hps_io_sdio_inst_D3
			hps_io_hps_io_usb1_inst_D0      => CONNECTED_TO_hps_io_hps_io_usb1_inst_D0,      --                     .hps_io_usb1_inst_D0
			hps_io_hps_io_usb1_inst_D1      => CONNECTED_TO_hps_io_hps_io_usb1_inst_D1,      --                     .hps_io_usb1_inst_D1
			hps_io_hps_io_usb1_inst_D2      => CONNECTED_TO_hps_io_hps_io_usb1_inst_D2,      --                     .hps_io_usb1_inst_D2
			hps_io_hps_io_usb1_inst_D3      => CONNECTED_TO_hps_io_hps_io_usb1_inst_D3,      --                     .hps_io_usb1_inst_D3
			hps_io_hps_io_usb1_inst_D4      => CONNECTED_TO_hps_io_hps_io_usb1_inst_D4,      --                     .hps_io_usb1_inst_D4
			hps_io_hps_io_usb1_inst_D5      => CONNECTED_TO_hps_io_hps_io_usb1_inst_D5,      --                     .hps_io_usb1_inst_D5
			hps_io_hps_io_usb1_inst_D6      => CONNECTED_TO_hps_io_hps_io_usb1_inst_D6,      --                     .hps_io_usb1_inst_D6
			hps_io_hps_io_usb1_inst_D7      => CONNECTED_TO_hps_io_hps_io_usb1_inst_D7,      --                     .hps_io_usb1_inst_D7
			hps_io_hps_io_usb1_inst_CLK     => CONNECTED_TO_hps_io_hps_io_usb1_inst_CLK,     --                     .hps_io_usb1_inst_CLK
			hps_io_hps_io_usb1_inst_STP     => CONNECTED_TO_hps_io_hps_io_usb1_inst_STP,     --                     .hps_io_usb1_inst_STP
			hps_io_hps_io_usb1_inst_DIR     => CONNECTED_TO_hps_io_hps_io_usb1_inst_DIR,     --                     .hps_io_usb1_inst_DIR
			hps_io_hps_io_usb1_inst_NXT     => CONNECTED_TO_hps_io_hps_io_usb1_inst_NXT,     --                     .hps_io_usb1_inst_NXT
			hps_io_hps_io_spim1_inst_CLK    => CONNECTED_TO_hps_io_hps_io_spim1_inst_CLK,    --                     .hps_io_spim1_inst_CLK
			hps_io_hps_io_spim1_inst_MOSI   => CONNECTED_TO_hps_io_hps_io_spim1_inst_MOSI,   --                     .hps_io_spim1_inst_MOSI
			hps_io_hps_io_spim1_inst_MISO   => CONNECTED_TO_hps_io_hps_io_spim1_inst_MISO,   --                     .hps_io_spim1_inst_MISO
			hps_io_hps_io_spim1_inst_SS0    => CONNECTED_TO_hps_io_hps_io_spim1_inst_SS0,    --                     .hps_io_spim1_inst_SS0
			hps_io_hps_io_uart0_inst_RX     => CONNECTED_TO_hps_io_hps_io_uart0_inst_RX,     --                     .hps_io_uart0_inst_RX
			hps_io_hps_io_uart0_inst_TX     => CONNECTED_TO_hps_io_hps_io_uart0_inst_TX,     --                     .hps_io_uart0_inst_TX
			hps_io_hps_io_i2c0_inst_SDA     => CONNECTED_TO_hps_io_hps_io_i2c0_inst_SDA,     --                     .hps_io_i2c0_inst_SDA
			hps_io_hps_io_i2c0_inst_SCL     => CONNECTED_TO_hps_io_hps_io_i2c0_inst_SCL,     --                     .hps_io_i2c0_inst_SCL
			hps_io_hps_io_i2c1_inst_SDA     => CONNECTED_TO_hps_io_hps_io_i2c1_inst_SDA,     --                     .hps_io_i2c1_inst_SDA
			hps_io_hps_io_i2c1_inst_SCL     => CONNECTED_TO_hps_io_hps_io_i2c1_inst_SCL,     --                     .hps_io_i2c1_inst_SCL
			hps_io_hps_io_gpio_inst_GPIO09  => CONNECTED_TO_hps_io_hps_io_gpio_inst_GPIO09,  --                     .hps_io_gpio_inst_GPIO09
			hps_io_hps_io_gpio_inst_GPIO35  => CONNECTED_TO_hps_io_hps_io_gpio_inst_GPIO35,  --                     .hps_io_gpio_inst_GPIO35
			hps_io_hps_io_gpio_inst_GPIO40  => CONNECTED_TO_hps_io_hps_io_gpio_inst_GPIO40,  --                     .hps_io_gpio_inst_GPIO40
			hps_io_hps_io_gpio_inst_GPIO41  => CONNECTED_TO_hps_io_hps_io_gpio_inst_GPIO41,  --                     .hps_io_gpio_inst_GPIO41
			hps_io_hps_io_gpio_inst_GPIO48  => CONNECTED_TO_hps_io_hps_io_gpio_inst_GPIO48,  --                     .hps_io_gpio_inst_GPIO48
			hps_io_hps_io_gpio_inst_GPIO53  => CONNECTED_TO_hps_io_hps_io_gpio_inst_GPIO53,  --                     .hps_io_gpio_inst_GPIO53
			hps_io_hps_io_gpio_inst_GPIO54  => CONNECTED_TO_hps_io_hps_io_gpio_inst_GPIO54,  --                     .hps_io_gpio_inst_GPIO54
			hps_io_hps_io_gpio_inst_GPIO61  => CONNECTED_TO_hps_io_hps_io_gpio_inst_GPIO61,  --                     .hps_io_gpio_inst_GPIO61
			irda_TXD                        => CONNECTED_TO_irda_TXD,                        --                 irda.TXD
			irda_RXD                        => CONNECTED_TO_irda_RXD,                        --                     .RXD
			leds_export                     => CONNECTED_TO_leds_export,                     --                 leds.export
			memory_mem_a                    => CONNECTED_TO_memory_mem_a,                    --               memory.mem_a
			memory_mem_ba                   => CONNECTED_TO_memory_mem_ba,                   --                     .mem_ba
			memory_mem_ck                   => CONNECTED_TO_memory_mem_ck,                   --                     .mem_ck
			memory_mem_ck_n                 => CONNECTED_TO_memory_mem_ck_n,                 --                     .mem_ck_n
			memory_mem_cke                  => CONNECTED_TO_memory_mem_cke,                  --                     .mem_cke
			memory_mem_cs_n                 => CONNECTED_TO_memory_mem_cs_n,                 --                     .mem_cs_n
			memory_mem_ras_n                => CONNECTED_TO_memory_mem_ras_n,                --                     .mem_ras_n
			memory_mem_cas_n                => CONNECTED_TO_memory_mem_cas_n,                --                     .mem_cas_n
			memory_mem_we_n                 => CONNECTED_TO_memory_mem_we_n,                 --                     .mem_we_n
			memory_mem_reset_n              => CONNECTED_TO_memory_mem_reset_n,              --                     .mem_reset_n
			memory_mem_dq                   => CONNECTED_TO_memory_mem_dq,                   --                     .mem_dq
			memory_mem_dqs                  => CONNECTED_TO_memory_mem_dqs,                  --                     .mem_dqs
			memory_mem_dqs_n                => CONNECTED_TO_memory_mem_dqs_n,                --                     .mem_dqs_n
			memory_mem_odt                  => CONNECTED_TO_memory_mem_odt,                  --                     .mem_odt
			memory_mem_dm                   => CONNECTED_TO_memory_mem_dm,                   --                     .mem_dm
			memory_oct_rzqin                => CONNECTED_TO_memory_oct_rzqin,                --                     .oct_rzqin
			ps2_port_CLK                    => CONNECTED_TO_ps2_port_CLK,                    --             ps2_port.CLK
			ps2_port_DAT                    => CONNECTED_TO_ps2_port_DAT,                    --                     .DAT
			ps2_port_dual_CLK               => CONNECTED_TO_ps2_port_dual_CLK,               --        ps2_port_dual.CLK
			ps2_port_dual_DAT               => CONNECTED_TO_ps2_port_dual_DAT,               --                     .DAT
			pushbuttons_export              => CONNECTED_TO_pushbuttons_export,              --          pushbuttons.export
			sdram_addr                      => CONNECTED_TO_sdram_addr,                      --                sdram.addr
			sdram_ba                        => CONNECTED_TO_sdram_ba,                        --                     .ba
			sdram_cas_n                     => CONNECTED_TO_sdram_cas_n,                     --                     .cas_n
			sdram_cke                       => CONNECTED_TO_sdram_cke,                       --                     .cke
			sdram_cs_n                      => CONNECTED_TO_sdram_cs_n,                      --                     .cs_n
			sdram_dq                        => CONNECTED_TO_sdram_dq,                        --                     .dq
			sdram_dqm                       => CONNECTED_TO_sdram_dqm,                       --                     .dqm
			sdram_ras_n                     => CONNECTED_TO_sdram_ras_n,                     --                     .ras_n
			sdram_we_n                      => CONNECTED_TO_sdram_we_n,                      --                     .we_n
			sdram_clk_clk                   => CONNECTED_TO_sdram_clk_clk,                   --            sdram_clk.clk
			slider_switches_export          => CONNECTED_TO_slider_switches_export,          --      slider_switches.export
			system_pll_ref_clk_clk          => CONNECTED_TO_system_pll_ref_clk_clk,          --   system_pll_ref_clk.clk
			system_pll_ref_reset_reset      => CONNECTED_TO_system_pll_ref_reset_reset,      -- system_pll_ref_reset.reset
			vga_CLK                         => CONNECTED_TO_vga_CLK,                         --                  vga.CLK
			vga_HS                          => CONNECTED_TO_vga_HS,                          --                     .HS
			vga_VS                          => CONNECTED_TO_vga_VS,                          --                     .VS
			vga_BLANK                       => CONNECTED_TO_vga_BLANK,                       --                     .BLANK
			vga_SYNC                        => CONNECTED_TO_vga_SYNC,                        --                     .SYNC
			vga_R                           => CONNECTED_TO_vga_R,                           --                     .R
			vga_G                           => CONNECTED_TO_vga_G,                           --                     .G
			vga_B                           => CONNECTED_TO_vga_B,                           --                     .B
			vga_pll_d5m_clk_clk             => CONNECTED_TO_vga_pll_d5m_clk_clk,             --      vga_pll_d5m_clk.clk
			vga_pll_ref_clk_clk             => CONNECTED_TO_vga_pll_ref_clk_clk,             --      vga_pll_ref_clk.clk
			vga_pll_ref_reset_reset         => CONNECTED_TO_vga_pll_ref_reset_reset,         --    vga_pll_ref_reset.reset
			video_in_PIXEL_CLK              => CONNECTED_TO_video_in_PIXEL_CLK,              --             video_in.PIXEL_CLK
			video_in_LINE_VALID             => CONNECTED_TO_video_in_LINE_VALID,             --                     .LINE_VALID
			video_in_FRAME_VALID            => CONNECTED_TO_video_in_FRAME_VALID,            --                     .FRAME_VALID
			video_in_pixel_clk_reset        => CONNECTED_TO_video_in_pixel_clk_reset,        --                     .pixel_clk_reset
			video_in_PIXEL_DATA             => CONNECTED_TO_video_in_PIXEL_DATA              --                     .PIXEL_DATA
		);

