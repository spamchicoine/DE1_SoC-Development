	component system is
		port (
			acl_iface_d5m_config_I2C_SDAT       : inout std_logic                     := 'X';             -- I2C_SDAT
			acl_iface_d5m_config_I2C_SCLK       : out   std_logic;                                        -- I2C_SCLK
			acl_iface_d5m_config_exposure       : in    std_logic_vector(15 downto 0) := (others => 'X'); -- exposure
			acl_iface_led_pio_export            : out   std_logic_vector(5 downto 0);                     -- export
			acl_iface_pushbuttons_export        : in    std_logic_vector(3 downto 0)  := (others => 'X'); -- export
			acl_iface_vga_CLK                   : out   std_logic;                                        -- CLK
			acl_iface_vga_HS                    : out   std_logic;                                        -- HS
			acl_iface_vga_VS                    : out   std_logic;                                        -- VS
			acl_iface_vga_BLANK                 : out   std_logic;                                        -- BLANK
			acl_iface_vga_SYNC                  : out   std_logic;                                        -- SYNC
			acl_iface_vga_R                     : out   std_logic_vector(7 downto 0);                     -- R
			acl_iface_vga_G                     : out   std_logic_vector(7 downto 0);                     -- G
			acl_iface_vga_B                     : out   std_logic_vector(7 downto 0);                     -- B
			acl_iface_vga_pll_d5m_clk_clk       : out   std_logic;                                        -- clk
			acl_iface_vga_pll_ref_clk_clk       : in    std_logic                     := 'X';             -- clk
			acl_iface_vga_pll_ref_reset_reset   : in    std_logic                     := 'X';             -- reset
			acl_iface_video_in_PIXEL_CLK        : in    std_logic                     := 'X';             -- PIXEL_CLK
			acl_iface_video_in_LINE_VALID       : in    std_logic                     := 'X';             -- LINE_VALID
			acl_iface_video_in_FRAME_VALID      : in    std_logic                     := 'X';             -- FRAME_VALID
			acl_iface_video_in_pixel_clk_reset  : in    std_logic                     := 'X';             -- pixel_clk_reset
			acl_iface_video_in_PIXEL_DATA       : in    std_logic_vector(11 downto 0) := (others => 'X'); -- PIXEL_DATA
			clk_50_clk                          : in    std_logic                     := 'X';             -- clk
			kernel_clk_clk                      : out   std_logic;                                        -- clk
			memory_mem_a                        : out   std_logic_vector(14 downto 0);                    -- mem_a
			memory_mem_ba                       : out   std_logic_vector(2 downto 0);                     -- mem_ba
			memory_mem_ck                       : out   std_logic;                                        -- mem_ck
			memory_mem_ck_n                     : out   std_logic;                                        -- mem_ck_n
			memory_mem_cke                      : out   std_logic;                                        -- mem_cke
			memory_mem_cs_n                     : out   std_logic;                                        -- mem_cs_n
			memory_mem_ras_n                    : out   std_logic;                                        -- mem_ras_n
			memory_mem_cas_n                    : out   std_logic;                                        -- mem_cas_n
			memory_mem_we_n                     : out   std_logic;                                        -- mem_we_n
			memory_mem_reset_n                  : out   std_logic;                                        -- mem_reset_n
			memory_mem_dq                       : inout std_logic_vector(31 downto 0) := (others => 'X'); -- mem_dq
			memory_mem_dqs                      : inout std_logic_vector(3 downto 0)  := (others => 'X'); -- mem_dqs
			memory_mem_dqs_n                    : inout std_logic_vector(3 downto 0)  := (others => 'X'); -- mem_dqs_n
			memory_mem_odt                      : out   std_logic;                                        -- mem_odt
			memory_mem_dm                       : out   std_logic_vector(3 downto 0);                     -- mem_dm
			memory_oct_rzqin                    : in    std_logic                     := 'X';             -- oct_rzqin
			peripheral_hps_io_emac1_inst_TX_CLK : out   std_logic;                                        -- hps_io_emac1_inst_TX_CLK
			peripheral_hps_io_emac1_inst_TXD0   : out   std_logic;                                        -- hps_io_emac1_inst_TXD0
			peripheral_hps_io_emac1_inst_TXD1   : out   std_logic;                                        -- hps_io_emac1_inst_TXD1
			peripheral_hps_io_emac1_inst_TXD2   : out   std_logic;                                        -- hps_io_emac1_inst_TXD2
			peripheral_hps_io_emac1_inst_TXD3   : out   std_logic;                                        -- hps_io_emac1_inst_TXD3
			peripheral_hps_io_emac1_inst_RXD0   : in    std_logic                     := 'X';             -- hps_io_emac1_inst_RXD0
			peripheral_hps_io_emac1_inst_MDIO   : inout std_logic                     := 'X';             -- hps_io_emac1_inst_MDIO
			peripheral_hps_io_emac1_inst_MDC    : out   std_logic;                                        -- hps_io_emac1_inst_MDC
			peripheral_hps_io_emac1_inst_RX_CTL : in    std_logic                     := 'X';             -- hps_io_emac1_inst_RX_CTL
			peripheral_hps_io_emac1_inst_TX_CTL : out   std_logic;                                        -- hps_io_emac1_inst_TX_CTL
			peripheral_hps_io_emac1_inst_RX_CLK : in    std_logic                     := 'X';             -- hps_io_emac1_inst_RX_CLK
			peripheral_hps_io_emac1_inst_RXD1   : in    std_logic                     := 'X';             -- hps_io_emac1_inst_RXD1
			peripheral_hps_io_emac1_inst_RXD2   : in    std_logic                     := 'X';             -- hps_io_emac1_inst_RXD2
			peripheral_hps_io_emac1_inst_RXD3   : in    std_logic                     := 'X';             -- hps_io_emac1_inst_RXD3
			peripheral_hps_io_sdio_inst_CMD     : inout std_logic                     := 'X';             -- hps_io_sdio_inst_CMD
			peripheral_hps_io_sdio_inst_D0      : inout std_logic                     := 'X';             -- hps_io_sdio_inst_D0
			peripheral_hps_io_sdio_inst_D1      : inout std_logic                     := 'X';             -- hps_io_sdio_inst_D1
			peripheral_hps_io_sdio_inst_CLK     : out   std_logic;                                        -- hps_io_sdio_inst_CLK
			peripheral_hps_io_sdio_inst_D2      : inout std_logic                     := 'X';             -- hps_io_sdio_inst_D2
			peripheral_hps_io_sdio_inst_D3      : inout std_logic                     := 'X';             -- hps_io_sdio_inst_D3
			peripheral_hps_io_usb1_inst_D0      : inout std_logic                     := 'X';             -- hps_io_usb1_inst_D0
			peripheral_hps_io_usb1_inst_D1      : inout std_logic                     := 'X';             -- hps_io_usb1_inst_D1
			peripheral_hps_io_usb1_inst_D2      : inout std_logic                     := 'X';             -- hps_io_usb1_inst_D2
			peripheral_hps_io_usb1_inst_D3      : inout std_logic                     := 'X';             -- hps_io_usb1_inst_D3
			peripheral_hps_io_usb1_inst_D4      : inout std_logic                     := 'X';             -- hps_io_usb1_inst_D4
			peripheral_hps_io_usb1_inst_D5      : inout std_logic                     := 'X';             -- hps_io_usb1_inst_D5
			peripheral_hps_io_usb1_inst_D6      : inout std_logic                     := 'X';             -- hps_io_usb1_inst_D6
			peripheral_hps_io_usb1_inst_D7      : inout std_logic                     := 'X';             -- hps_io_usb1_inst_D7
			peripheral_hps_io_usb1_inst_CLK     : in    std_logic                     := 'X';             -- hps_io_usb1_inst_CLK
			peripheral_hps_io_usb1_inst_STP     : out   std_logic;                                        -- hps_io_usb1_inst_STP
			peripheral_hps_io_usb1_inst_DIR     : in    std_logic                     := 'X';             -- hps_io_usb1_inst_DIR
			peripheral_hps_io_usb1_inst_NXT     : in    std_logic                     := 'X';             -- hps_io_usb1_inst_NXT
			peripheral_hps_io_uart0_inst_RX     : in    std_logic                     := 'X';             -- hps_io_uart0_inst_RX
			peripheral_hps_io_uart0_inst_TX     : out   std_logic;                                        -- hps_io_uart0_inst_TX
			peripheral_hps_io_i2c0_inst_SDA     : inout std_logic                     := 'X';             -- hps_io_i2c0_inst_SDA
			peripheral_hps_io_i2c0_inst_SCL     : inout std_logic                     := 'X';             -- hps_io_i2c0_inst_SCL
			peripheral_hps_io_gpio_inst_GPIO53  : inout std_logic                     := 'X';             -- hps_io_gpio_inst_GPIO53
			reset_50_reset_n                    : in    std_logic                     := 'X';             -- reset_n
			acl_iface_adc_sclk                  : out   std_logic;                                        -- sclk
			acl_iface_adc_cs_n                  : out   std_logic;                                        -- cs_n
			acl_iface_adc_dout                  : in    std_logic                     := 'X';             -- dout
			acl_iface_adc_din                   : out   std_logic;                                        -- din
			acl_iface_av_config_SDAT            : inout std_logic                     := 'X';             -- SDAT
			acl_iface_av_config_SCLK            : out   std_logic                                         -- SCLK
		);
	end component system;

	u0 : component system
		port map (
			acl_iface_d5m_config_I2C_SDAT       => CONNECTED_TO_acl_iface_d5m_config_I2C_SDAT,       --        acl_iface_d5m_config.I2C_SDAT
			acl_iface_d5m_config_I2C_SCLK       => CONNECTED_TO_acl_iface_d5m_config_I2C_SCLK,       --                            .I2C_SCLK
			acl_iface_d5m_config_exposure       => CONNECTED_TO_acl_iface_d5m_config_exposure,       --                            .exposure
			acl_iface_led_pio_export            => CONNECTED_TO_acl_iface_led_pio_export,            --           acl_iface_led_pio.export
			acl_iface_pushbuttons_export        => CONNECTED_TO_acl_iface_pushbuttons_export,        --       acl_iface_pushbuttons.export
			acl_iface_vga_CLK                   => CONNECTED_TO_acl_iface_vga_CLK,                   --               acl_iface_vga.CLK
			acl_iface_vga_HS                    => CONNECTED_TO_acl_iface_vga_HS,                    --                            .HS
			acl_iface_vga_VS                    => CONNECTED_TO_acl_iface_vga_VS,                    --                            .VS
			acl_iface_vga_BLANK                 => CONNECTED_TO_acl_iface_vga_BLANK,                 --                            .BLANK
			acl_iface_vga_SYNC                  => CONNECTED_TO_acl_iface_vga_SYNC,                  --                            .SYNC
			acl_iface_vga_R                     => CONNECTED_TO_acl_iface_vga_R,                     --                            .R
			acl_iface_vga_G                     => CONNECTED_TO_acl_iface_vga_G,                     --                            .G
			acl_iface_vga_B                     => CONNECTED_TO_acl_iface_vga_B,                     --                            .B
			acl_iface_vga_pll_d5m_clk_clk       => CONNECTED_TO_acl_iface_vga_pll_d5m_clk_clk,       --   acl_iface_vga_pll_d5m_clk.clk
			acl_iface_vga_pll_ref_clk_clk       => CONNECTED_TO_acl_iface_vga_pll_ref_clk_clk,       --   acl_iface_vga_pll_ref_clk.clk
			acl_iface_vga_pll_ref_reset_reset   => CONNECTED_TO_acl_iface_vga_pll_ref_reset_reset,   -- acl_iface_vga_pll_ref_reset.reset
			acl_iface_video_in_PIXEL_CLK        => CONNECTED_TO_acl_iface_video_in_PIXEL_CLK,        --          acl_iface_video_in.PIXEL_CLK
			acl_iface_video_in_LINE_VALID       => CONNECTED_TO_acl_iface_video_in_LINE_VALID,       --                            .LINE_VALID
			acl_iface_video_in_FRAME_VALID      => CONNECTED_TO_acl_iface_video_in_FRAME_VALID,      --                            .FRAME_VALID
			acl_iface_video_in_pixel_clk_reset  => CONNECTED_TO_acl_iface_video_in_pixel_clk_reset,  --                            .pixel_clk_reset
			acl_iface_video_in_PIXEL_DATA       => CONNECTED_TO_acl_iface_video_in_PIXEL_DATA,       --                            .PIXEL_DATA
			clk_50_clk                          => CONNECTED_TO_clk_50_clk,                          --                      clk_50.clk
			kernel_clk_clk                      => CONNECTED_TO_kernel_clk_clk,                      --                  kernel_clk.clk
			memory_mem_a                        => CONNECTED_TO_memory_mem_a,                        --                      memory.mem_a
			memory_mem_ba                       => CONNECTED_TO_memory_mem_ba,                       --                            .mem_ba
			memory_mem_ck                       => CONNECTED_TO_memory_mem_ck,                       --                            .mem_ck
			memory_mem_ck_n                     => CONNECTED_TO_memory_mem_ck_n,                     --                            .mem_ck_n
			memory_mem_cke                      => CONNECTED_TO_memory_mem_cke,                      --                            .mem_cke
			memory_mem_cs_n                     => CONNECTED_TO_memory_mem_cs_n,                     --                            .mem_cs_n
			memory_mem_ras_n                    => CONNECTED_TO_memory_mem_ras_n,                    --                            .mem_ras_n
			memory_mem_cas_n                    => CONNECTED_TO_memory_mem_cas_n,                    --                            .mem_cas_n
			memory_mem_we_n                     => CONNECTED_TO_memory_mem_we_n,                     --                            .mem_we_n
			memory_mem_reset_n                  => CONNECTED_TO_memory_mem_reset_n,                  --                            .mem_reset_n
			memory_mem_dq                       => CONNECTED_TO_memory_mem_dq,                       --                            .mem_dq
			memory_mem_dqs                      => CONNECTED_TO_memory_mem_dqs,                      --                            .mem_dqs
			memory_mem_dqs_n                    => CONNECTED_TO_memory_mem_dqs_n,                    --                            .mem_dqs_n
			memory_mem_odt                      => CONNECTED_TO_memory_mem_odt,                      --                            .mem_odt
			memory_mem_dm                       => CONNECTED_TO_memory_mem_dm,                       --                            .mem_dm
			memory_oct_rzqin                    => CONNECTED_TO_memory_oct_rzqin,                    --                            .oct_rzqin
			peripheral_hps_io_emac1_inst_TX_CLK => CONNECTED_TO_peripheral_hps_io_emac1_inst_TX_CLK, --                  peripheral.hps_io_emac1_inst_TX_CLK
			peripheral_hps_io_emac1_inst_TXD0   => CONNECTED_TO_peripheral_hps_io_emac1_inst_TXD0,   --                            .hps_io_emac1_inst_TXD0
			peripheral_hps_io_emac1_inst_TXD1   => CONNECTED_TO_peripheral_hps_io_emac1_inst_TXD1,   --                            .hps_io_emac1_inst_TXD1
			peripheral_hps_io_emac1_inst_TXD2   => CONNECTED_TO_peripheral_hps_io_emac1_inst_TXD2,   --                            .hps_io_emac1_inst_TXD2
			peripheral_hps_io_emac1_inst_TXD3   => CONNECTED_TO_peripheral_hps_io_emac1_inst_TXD3,   --                            .hps_io_emac1_inst_TXD3
			peripheral_hps_io_emac1_inst_RXD0   => CONNECTED_TO_peripheral_hps_io_emac1_inst_RXD0,   --                            .hps_io_emac1_inst_RXD0
			peripheral_hps_io_emac1_inst_MDIO   => CONNECTED_TO_peripheral_hps_io_emac1_inst_MDIO,   --                            .hps_io_emac1_inst_MDIO
			peripheral_hps_io_emac1_inst_MDC    => CONNECTED_TO_peripheral_hps_io_emac1_inst_MDC,    --                            .hps_io_emac1_inst_MDC
			peripheral_hps_io_emac1_inst_RX_CTL => CONNECTED_TO_peripheral_hps_io_emac1_inst_RX_CTL, --                            .hps_io_emac1_inst_RX_CTL
			peripheral_hps_io_emac1_inst_TX_CTL => CONNECTED_TO_peripheral_hps_io_emac1_inst_TX_CTL, --                            .hps_io_emac1_inst_TX_CTL
			peripheral_hps_io_emac1_inst_RX_CLK => CONNECTED_TO_peripheral_hps_io_emac1_inst_RX_CLK, --                            .hps_io_emac1_inst_RX_CLK
			peripheral_hps_io_emac1_inst_RXD1   => CONNECTED_TO_peripheral_hps_io_emac1_inst_RXD1,   --                            .hps_io_emac1_inst_RXD1
			peripheral_hps_io_emac1_inst_RXD2   => CONNECTED_TO_peripheral_hps_io_emac1_inst_RXD2,   --                            .hps_io_emac1_inst_RXD2
			peripheral_hps_io_emac1_inst_RXD3   => CONNECTED_TO_peripheral_hps_io_emac1_inst_RXD3,   --                            .hps_io_emac1_inst_RXD3
			peripheral_hps_io_sdio_inst_CMD     => CONNECTED_TO_peripheral_hps_io_sdio_inst_CMD,     --                            .hps_io_sdio_inst_CMD
			peripheral_hps_io_sdio_inst_D0      => CONNECTED_TO_peripheral_hps_io_sdio_inst_D0,      --                            .hps_io_sdio_inst_D0
			peripheral_hps_io_sdio_inst_D1      => CONNECTED_TO_peripheral_hps_io_sdio_inst_D1,      --                            .hps_io_sdio_inst_D1
			peripheral_hps_io_sdio_inst_CLK     => CONNECTED_TO_peripheral_hps_io_sdio_inst_CLK,     --                            .hps_io_sdio_inst_CLK
			peripheral_hps_io_sdio_inst_D2      => CONNECTED_TO_peripheral_hps_io_sdio_inst_D2,      --                            .hps_io_sdio_inst_D2
			peripheral_hps_io_sdio_inst_D3      => CONNECTED_TO_peripheral_hps_io_sdio_inst_D3,      --                            .hps_io_sdio_inst_D3
			peripheral_hps_io_usb1_inst_D0      => CONNECTED_TO_peripheral_hps_io_usb1_inst_D0,      --                            .hps_io_usb1_inst_D0
			peripheral_hps_io_usb1_inst_D1      => CONNECTED_TO_peripheral_hps_io_usb1_inst_D1,      --                            .hps_io_usb1_inst_D1
			peripheral_hps_io_usb1_inst_D2      => CONNECTED_TO_peripheral_hps_io_usb1_inst_D2,      --                            .hps_io_usb1_inst_D2
			peripheral_hps_io_usb1_inst_D3      => CONNECTED_TO_peripheral_hps_io_usb1_inst_D3,      --                            .hps_io_usb1_inst_D3
			peripheral_hps_io_usb1_inst_D4      => CONNECTED_TO_peripheral_hps_io_usb1_inst_D4,      --                            .hps_io_usb1_inst_D4
			peripheral_hps_io_usb1_inst_D5      => CONNECTED_TO_peripheral_hps_io_usb1_inst_D5,      --                            .hps_io_usb1_inst_D5
			peripheral_hps_io_usb1_inst_D6      => CONNECTED_TO_peripheral_hps_io_usb1_inst_D6,      --                            .hps_io_usb1_inst_D6
			peripheral_hps_io_usb1_inst_D7      => CONNECTED_TO_peripheral_hps_io_usb1_inst_D7,      --                            .hps_io_usb1_inst_D7
			peripheral_hps_io_usb1_inst_CLK     => CONNECTED_TO_peripheral_hps_io_usb1_inst_CLK,     --                            .hps_io_usb1_inst_CLK
			peripheral_hps_io_usb1_inst_STP     => CONNECTED_TO_peripheral_hps_io_usb1_inst_STP,     --                            .hps_io_usb1_inst_STP
			peripheral_hps_io_usb1_inst_DIR     => CONNECTED_TO_peripheral_hps_io_usb1_inst_DIR,     --                            .hps_io_usb1_inst_DIR
			peripheral_hps_io_usb1_inst_NXT     => CONNECTED_TO_peripheral_hps_io_usb1_inst_NXT,     --                            .hps_io_usb1_inst_NXT
			peripheral_hps_io_uart0_inst_RX     => CONNECTED_TO_peripheral_hps_io_uart0_inst_RX,     --                            .hps_io_uart0_inst_RX
			peripheral_hps_io_uart0_inst_TX     => CONNECTED_TO_peripheral_hps_io_uart0_inst_TX,     --                            .hps_io_uart0_inst_TX
			peripheral_hps_io_i2c0_inst_SDA     => CONNECTED_TO_peripheral_hps_io_i2c0_inst_SDA,     --                            .hps_io_i2c0_inst_SDA
			peripheral_hps_io_i2c0_inst_SCL     => CONNECTED_TO_peripheral_hps_io_i2c0_inst_SCL,     --                            .hps_io_i2c0_inst_SCL
			peripheral_hps_io_gpio_inst_GPIO53  => CONNECTED_TO_peripheral_hps_io_gpio_inst_GPIO53,  --                            .hps_io_gpio_inst_GPIO53
			reset_50_reset_n                    => CONNECTED_TO_reset_50_reset_n,                    --                    reset_50.reset_n
			acl_iface_adc_sclk                  => CONNECTED_TO_acl_iface_adc_sclk,                  --               acl_iface_adc.sclk
			acl_iface_adc_cs_n                  => CONNECTED_TO_acl_iface_adc_cs_n,                  --                            .cs_n
			acl_iface_adc_dout                  => CONNECTED_TO_acl_iface_adc_dout,                  --                            .dout
			acl_iface_adc_din                   => CONNECTED_TO_acl_iface_adc_din,                   --                            .din
			acl_iface_av_config_SDAT            => CONNECTED_TO_acl_iface_av_config_SDAT,            --         acl_iface_av_config.SDAT
			acl_iface_av_config_SCLK            => CONNECTED_TO_acl_iface_av_config_SCLK             --                            .SCLK
		);

